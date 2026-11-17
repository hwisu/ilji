//! Embeddings through an OpenAI-compatible endpoint (llama-server, Ollama, LM Studio, ...).
//! Only dialog chunks are embedded. Vectors are normalized, stored per model and chunk-text hash,
//! and searched exhaustively: a few tens of thousands of vectors take milliseconds.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Context;
use rusqlite::params;
use serde_json::{Value, json};

use crate::common::head_chars;
use crate::config::{Config, EmbedCfg, Precision};
use crate::store::Store;

pub struct Client {
    agent: ureq::Agent,
    url: String,
    model: String,
}

impl Client {
    pub fn new(cfg: &EmbedCfg, timeout: Duration) -> Client {
        let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(timeout)).build().into();
        Client { agent, url: cfg.url.clone(), model: cfg.model.clone() }
    }

    /// One request; returns normalized vectors in input order.
    pub fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut resp = self
            .agent
            .post(&self.url)
            .send_json(json!({ "model": self.model, "input": texts }))
            .with_context(|| format!("embedding endpoint {}", self.url))?;
        let v: Value = resp.body_mut().with_config().limit(512 * 1024 * 1024).read_json()?;
        let data = v.get("data").and_then(Value::as_array).context("embedding response has no data")?;
        let mut out = vec![Vec::new(); texts.len()];
        for (pos, d) in data.iter().enumerate() {
            let i = d.get("index").and_then(Value::as_u64).map_or(pos, |i| i as usize);
            let e = d.get("embedding").and_then(Value::as_array).context("embedding missing")?;
            let mut x: Vec<f32> = e.iter().map(|f| f.as_f64().unwrap_or(0.0) as f32).collect();
            normalize(&mut x);
            if i < out.len() {
                out[i] = x;
            }
        }
        anyhow::ensure!(out.iter().all(|x| !x.is_empty()), "the endpoint returned fewer vectors than texts");
        Ok(out)
    }
}

/// The name vectors are stored under: the model plus every setting that changes the embedded
/// text, so changing one re-embeds instead of mixing vectors of different texts.
pub fn key(ec: &EmbedCfg) -> String {
    let mut k = ec.model.clone();
    if ec.skip_imported_tools {
        k += "+noimported";
    }
    if !ec.passage_prefix.is_empty() {
        k += &format!("+p{}", &crate::store::text_hash(&ec.passage_prefix)[..8]);
    }
    k
}

static IMPORTED: std::sync::LazyLock<[regex::Regex; 2]> = std::sync::LazyLock::new(|| {
    ["call", "result"].map(|k| {
        regex::Regex::new(&format!(r"(?s)\[external_agent_tool_{k}[^\]\n]*\].*?(?:\[/external_agent_tool_{k}\]|\z)")).unwrap()
    })
});

/// Remove imported tool transcripts, including one cut off at the start of the chunk.
pub fn strip_imported_tools(text: &str) -> String {
    let mut t = text;
    let close = t.find("[/external_agent_tool_");
    let open = t.find("[external_agent_tool_");
    if let Some(c) = close {
        if open.is_none_or(|o| c < o) {
            let end = t[c..].find(']').map_or(t.len(), |e| c + e + 1);
            t = &t[end..];
        }
    }
    let mut out = t.to_string();
    for re in IMPORTED.iter() {
        out = re.replace_all(&out, " ").into_owned();
    }
    out
}

/// The text sent for a chunk, or None when too little is left to embed.
pub fn passage_text(ec: &EmbedCfg, text: &str) -> Option<String> {
    let body = if ec.skip_imported_tools { strip_imported_tools(text) } else { text.to_string() };
    let body = body.trim();
    if body.chars().filter(|c| !c.is_whitespace()).count() < ec.min_chars {
        return None;
    }
    Some(format!("{}{}", ec.passage_prefix, head_chars(body, ec.max_chars)))
}

pub fn normalize(x: &mut [f32]) {
    let n = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    if n > 0.0 {
        x.iter_mut().for_each(|v| *v /= n);
    }
}

/// Encode a normalized vector: (scale, bytes).
pub fn encode(x: &[f32], p: Precision) -> (f32, Vec<u8>) {
    match p {
        Precision::F32 => (1.0, x.iter().flat_map(|v| v.to_le_bytes()).collect()),
        Precision::I8 => {
            let max = x.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-12);
            let scale = max / 127.0;
            (scale, x.iter().map(|v| (v / scale).round().clamp(-127.0, 127.0) as i8 as u8).collect())
        }
    }
}

/// All vectors of one model, flattened for a fast scan.
pub struct Vectors {
    pub dim: usize,
    pub hashes: Vec<String>,
    /// Per row: scale, and whether it is i8 (else f32), and its offset in the matching buffer.
    rows: Vec<(f32, bool, usize)>,
    i8s: Vec<i8>,
    f32s: Vec<f32>,
}

impl Vectors {
    pub fn load(store: &Store, model: &str) -> anyhow::Result<Vectors> {
        let mut v = Vectors { dim: 0, hashes: vec![], rows: vec![], i8s: vec![], f32s: vec![] };
        let mut q = store.conn.prepare("SELECT hash, dim, scale, data FROM vectors WHERE model=?1")?;
        let mut rows = q.query([model])?;
        while let Some(r) = rows.next()? {
            let dim: usize = r.get::<_, i64>(1)? as usize;
            if dim == 0 {
                continue; // a chunk with too little text: handled, nothing to search
            }
            if v.dim == 0 {
                v.dim = dim;
            }
            if dim != v.dim {
                continue;
            }
            let data: Vec<u8> = r.get(3)?;
            let scale: f64 = r.get(2)?;
            if data.len() == dim {
                v.rows.push((scale as f32, true, v.i8s.len()));
                v.i8s.extend(data.iter().map(|b| *b as i8));
            } else if data.len() == dim * 4 {
                v.rows.push((1.0, false, v.f32s.len()));
                v.f32s.extend(data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])));
            } else {
                continue;
            }
            v.hashes.push(r.get(0)?);
        }
        Ok(v)
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// The `n` most similar rows: (row, cosine).
    pub fn top(&self, q: &[f32], n: usize) -> Vec<(usize, f32)> {
        if q.len() != self.dim {
            return vec![];
        }
        let mut scored: Vec<(usize, f32)> = self
            .rows
            .iter()
            .enumerate()
            .map(|(i, &(scale, is_i8, off))| {
                let s = if is_i8 {
                    self.i8s[off..off + self.dim].iter().zip(q).map(|(a, b)| *a as f32 * b).sum::<f32>() * scale
                } else {
                    self.f32s[off..off + self.dim].iter().zip(q).map(|(a, b)| a * b).sum::<f32>()
                };
                (i, s)
            })
            .collect();
        let n = n.min(scored.len());
        if n == 0 {
            return vec![];
        }
        scored.select_nth_unstable_by(n - 1, |a, b| b.1.total_cmp(&a.1));
        scored.truncate(n);
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored
    }
}

static CACHE: Mutex<Option<(String, i64, Arc<Vectors>)>> = Mutex::new(None);

/// Vectors of `model`, reloaded only when their count changed (cheap for a long-running server).
pub fn cached(store: &Store, model: &str) -> anyhow::Result<Arc<Vectors>> {
    let count: i64 = store.conn.query_row("SELECT count(*) FROM vectors WHERE model=?1", [model], |r| r.get(0))?;
    let mut guard = CACHE.lock().unwrap();
    if let Some((m, c, v)) = guard.as_ref() {
        if m == model && *c == count {
            return Ok(v.clone());
        }
    }
    let v = Arc::new(Vectors::load(store, model)?);
    *guard = Some((model.to_string(), count, v.clone()));
    Ok(v)
}

pub fn query_vector(cfg: &Config, text: &str) -> anyhow::Result<Vec<f32>> {
    let client = Client::new(&cfg.embed, Duration::from_secs(cfg.embed.timeout_secs.min(30)));
    Ok(client.embed(&[format!("{}{text}", cfg.embed.query_prefix)])?.remove(0))
}

#[derive(Debug, Default)]
pub struct Report {
    pub todo: usize,
    pub done: usize,
    pub chars: usize,
    pub secs: f64,
}

/// Dialog chunks that have no vector for the configured model yet: (hash, text), shortest first.
pub fn pending(store: &Store, cfg: &Config) -> anyhow::Result<Vec<(String, String)>> {
    let mut q = store.conn.prepare(
        "SELECT c.hash, min(c.text) FROM chunks c
         WHERE c.kind = 'dialog' AND c.hash IS NOT NULL
           AND NOT EXISTS (SELECT 1 FROM vectors v WHERE v.model = ?1 AND v.hash = c.hash)
         GROUP BY c.hash",
    )?;
    let mut todo: Vec<(String, String)> = q
        .query_map([key(&cfg.embed)], |r| Ok((r.get::<_, String>(0)?, r.get::<_, Option<String>>(1)?.unwrap_or_default())))?
        .flatten()
        .collect();
    todo.sort_by_key(|(_, t)| t.len());
    Ok(todo)
}

/// Embed what is pending, in batches of similar length. `progress(done, total, chars_per_sec)`.
pub fn backfill(
    store: &Store,
    cfg: &Config,
    limit: Option<usize>,
    mut progress: impl FnMut(usize, usize, f64),
) -> anyhow::Result<Report> {
    let ec = &cfg.embed;
    let mut todo = pending(store, cfg)?;
    if let Some(l) = limit {
        todo.truncate(l);
    }
    let mut rep = Report { todo: todo.len(), ..Default::default() };
    let client = Client::new(ec, Duration::from_secs(ec.timeout_secs));
    let name = key(ec);
    let t0 = Instant::now();
    for batch in todo.chunks(ec.batch.max(1)) {
        let texts: Vec<Option<String>> = batch.iter().map(|(_, t)| passage_text(ec, t)).collect();
        let send: Vec<String> = texts.iter().flatten().cloned().collect();
        let mut vecs = if send.is_empty() { vec![] } else { client.embed(&send)? }.into_iter();
        let tx = store.conn.unchecked_transaction()?;
        {
            let mut ins = tx.prepare_cached("INSERT OR REPLACE INTO vectors(model, hash, dim, scale, data) VALUES(?1, ?2, ?3, ?4, ?5)")?;
            for ((hash, _), text) in batch.iter().zip(&texts) {
                match text {
                    Some(_) => {
                        let v = vecs.next().context("vector count mismatch")?;
                        let (scale, data) = encode(&v, ec.precision);
                        ins.execute(params![name, hash, v.len() as i64, scale as f64, data])?;
                    }
                    None => {
                        ins.execute(params![name, hash, 0i64, 0.0f64, Vec::<u8>::new()])?;
                    }
                }
            }
        }
        tx.commit()?;
        rep.done += batch.len();
        rep.chars += send.iter().map(|t| t.chars().count()).sum::<usize>();
        let secs = t0.elapsed().as_secs_f64().max(1e-3);
        progress(rep.done, rep.todo, rep.chars as f64 / secs);
    }
    rep.secs = t0.elapsed().as_secs_f64();
    Ok(rep)
}

/// Whether the endpoint answers within a few seconds.
pub fn reachable(cfg: &EmbedCfg) -> bool {
    Client::new(cfg, Duration::from_secs(5)).embed(&["ping".to_string()]).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_imported_tool_blocks() {
        let t = "cut tail of an output\n[/external_agent_tool_result]\nI ran the tests.\n[external_agent_tool_call: Bash]\ncargo test\n[/external_agent_tool_call]\n[external_agent_tool_result]\n3 passed\n[/external_agent_tool_result]\nAll green. [external_agent_tool_call: Read]\nsrc/main.rs";
        let s = strip_imported_tools(t);
        assert!(s.contains("I ran the tests.") && s.contains("All green."), "{s}");
        assert!(!s.contains("cargo test") && !s.contains("3 passed") && !s.contains("cut tail") && !s.contains("src/main.rs"), "{s}");
        let ec = EmbedCfg::default();
        assert!(passage_text(&ec, "[external_agent_tool_call: Bash]\nls\n[/external_agent_tool_call]").is_none());
    }

    #[test]
    fn i8_keeps_cosine() {
        let mut a: Vec<f32> = (0..64).map(|i| ((i * 7 % 13) as f32 - 6.0) / 3.0).collect();
        let mut b: Vec<f32> = (0..64).map(|i| ((i * 5 % 11) as f32 - 5.0) / 2.0).collect();
        normalize(&mut a);
        normalize(&mut b);
        let exact: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let (scale, bytes) = encode(&b, Precision::I8);
        let approx: f32 = bytes.iter().zip(&a).map(|(q, x)| *q as i8 as f32 * x).sum::<f32>() * scale;
        assert!((exact - approx).abs() < 0.01, "{exact} vs {approx}");
    }
}
