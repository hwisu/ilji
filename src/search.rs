//! Keyword search over chunks (FTS5 BM25), grouping, snippets, and the lookups built on it:
//! open, recent, files, fix.

use std::collections::{HashMap, HashSet};

use rusqlite::{OptionalExtension, params};
use serde::Serialize;

use crate::chunk::{char_slice, tool_text};
use crate::config::{Config, Fusion, Group, Hybrid, Mode, now};
use crate::store::Store;
use crate::tokenize::{KoreanOpts, fts_query, query_terms, words};

#[derive(Debug, Clone, Default)]
pub struct Query {
    pub text: String,
    pub k: usize,
    pub group: Group,
    pub per_session: usize,
    pub source: Option<String>,
    pub repo: Option<String>,
    pub kind: Option<String>,
    pub exclude_ids: Vec<i64>,
    pub include_recent: bool,
    pub since: Option<f64>,
    pub ko: KoreanOpts,
    /// A ready FTS5 expression instead of `text` (used by `files`).
    pub raw_match: Option<String>,
}

impl Default for Group {
    fn default() -> Self {
        Group::Session
    }
}

impl Query {
    pub fn new(text: &str, cfg: &Config) -> Query {
        let s = &cfg.search;
        Query {
            text: text.to_string(),
            k: s.k,
            group: s.group,
            per_session: s.per_session,
            ko: KoreanOpts { stem: s.korean_stem, prefix: s.korean_prefix },
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Hit {
    pub chunk: i64,
    pub session: String,
    pub source: String,
    pub title: String,
    pub repo: String,
    pub ts: f64,
    pub kind: String,
    pub event: i64,
    pub score: f64,
    #[serde(skip)]
    pub text: String,
    pub snippet: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionHits {
    pub session: String,
    pub source: String,
    pub title: String,
    pub repo: String,
    pub started_at: f64,
    pub hits: Vec<Hit>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Results {
    pub query: String,
    /// keyword, semantic or hybrid: what actually ran.
    pub mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    pub matched: usize,
    pub sessions: Vec<SessionHits>,
}

/// Search by `search.mode`. Chunk mode returns one session entry per hit, so rendering stays uniform.
pub fn search(store: &Store, cfg: &Config, q: &Query) -> anyhow::Result<Results> {
    let s = &cfg.search;
    let lexical = s.mode == Mode::Hybrid && s.hybrid == Hybrid::Auto && is_lexical(&q.text);
    let want_dense = cfg.embed.enabled
        && s.mode != Mode::Keyword
        && !lexical
        && q.raw_match.is_none()
        && q.kind.as_deref() != Some("tool");
    let mut note = None;
    let dense = if want_dense {
        match dense_lane(store, cfg, q) {
            Ok(v) => Some(v),
            Err(e) => {
                note = Some(format!("embeddings unavailable ({e:#}); keyword results only"));
                None
            }
        }
    } else {
        None
    };
    let keyword = if s.mode == Mode::Semantic && dense.is_some() { vec![] } else { keyword_lane(store, cfg, q)? };
    if keyword.is_empty() && dense.is_none() && q.raw_match.is_none() && fts_query(&q.text, q.ko).is_empty() {
        anyhow::bail!("the query has no searchable words");
    }
    let (ranked, mode) = match dense {
        None if lexical => (keyword, "hybrid:keyword"),
        None => (keyword, "keyword"),
        Some(d) if s.mode == Mode::Semantic => (d, "semantic"),
        Some(d) => (
            match s.hybrid {
                Hybrid::Rrf | Hybrid::Auto => rrf(keyword, d, s, false),
                Hybrid::Balanced => rrf(keyword, d, s, true),
                Hybrid::Slots => slots(keyword, d, s),
            },
            "hybrid",
        ),
    };
    let mut r = finish(store, cfg, q, ranked)?;
    r.note = note;
    r.mode = mode.to_string();
    Ok(r)
}

/// Whether a query reads like a lookup rather than a question: it names an identifier (a path,
/// snake_case, a dotted or numbered token, CamelCase, something quoted) or is a few words without
/// a question mark. Such queries are served best by keyword search alone.
pub fn is_lexical(text: &str) -> bool {
    let t = text.trim();
    if t.contains('`') || t.contains('"') {
        return true;
    }
    let words: Vec<&str> = t.split_whitespace().collect();
    let ident = |w: &str| {
        // Sentence punctuation around a word is not part of it; a dot or colon inside one is.
        let w = w.trim_matches(|c: char| !c.is_alphanumeric() && !matches!(c, '/' | '_' | '#'));
        let ascii = w.is_ascii() && w.chars().any(|c| c.is_ascii_alphabetic());
        ascii
            && (w.contains(['_', '/', '.', ':', '#', '='])
                || (w.chars().any(|c| c.is_ascii_digit()) && w.chars().any(|c| c.is_ascii_alphabetic()))
                || (w.chars().skip(1).any(|c| c.is_ascii_uppercase()) && w.chars().any(|c| c.is_ascii_lowercase())))
    };
    words.iter().any(|w| ident(w)) || (words.len() <= 5 && !t.ends_with('?'))
}

/// Reciprocal rank fusion. With `balance_tools`, a tool chunk (which the embedding lane cannot
/// return) has its keyword term scaled up to what a chunk found by both lanes can reach.
fn rrf(keyword: Vec<Hit>, dense: Vec<Hit>, s: &crate::config::SearchCfg, balance_tools: bool) -> Vec<Hit> {
    let boost = (s.keyword_weight + s.dense_weight) / s.keyword_weight.max(1e-9);
    let mut fused: HashMap<i64, (f64, Hit)> = HashMap::new();
    for (rank, h) in keyword.into_iter().enumerate() {
        let w = match (h.kind == "tool", balance_tools) {
            (true, true) => s.keyword_weight * boost,
            (true, false) => s.keyword_weight * s.tool_boost,
            _ => s.keyword_weight,
        };
        fused.entry(h.chunk).or_insert((0.0, h)).0 += w / (s.rrf_k + rank as f64 + 1.0);
    }
    for (rank, h) in dense.into_iter().enumerate() {
        fused.entry(h.chunk).or_insert((0.0, h)).0 += s.dense_weight / (s.rrf_k + rank as f64 + 1.0);
    }
    let mut v: Vec<(f64, Hit)> = fused.into_values().collect();
    v.sort_by(|a, b| b.0.total_cmp(&a.0));
    v.into_iter()
        .map(|(score, mut h)| {
            h.score = score * 100.0;
            h
        })
        .collect()
}

/// Tool chunks keep their keyword positions; the dialog positions are refilled from the fusion of
/// keyword dialog hits and embedding hits.
fn slots(keyword: Vec<Hit>, dense: Vec<Hit>, s: &crate::config::SearchCfg) -> Vec<Hit> {
    let dialog: Vec<Hit> = keyword.iter().filter(|h| h.kind != "tool").cloned().collect();
    let mut fused = rrf(dialog, dense, s, false).into_iter();
    let mut out = Vec::with_capacity(keyword.len());
    for h in keyword {
        if h.kind == "tool" {
            out.push(h);
        } else if let Some(d) = fused.next() {
            out.push(d);
        }
    }
    out.extend(fused);
    out
}

/// Embedding lane: the nearest dialog chunks that pass the query's filters, most similar first.
fn dense_lane(store: &Store, cfg: &Config, q: &Query) -> anyhow::Result<Vec<Hit>> {
    let vecs = crate::embed::cached(store, &crate::embed::key(&cfg.embed))?;
    anyhow::ensure!(vecs.len() > 0, "no vectors for {} yet; run `ilji embed`", cfg.embed.model);
    let qv = crate::embed::query_vector(cfg, &q.text)?;
    let top = vecs.top(&qv, cfg.search.dense_k.max(q.k));
    if top.is_empty() {
        return Ok(vec![]);
    }
    let rank: HashMap<&str, (usize, f32)> =
        top.iter().enumerate().map(|(r, (i, s))| (vecs.hashes[*i].as_str(), (r, *s))).collect();
    let n = top.len();
    let marks = (1..=n).map(|i| format!("?{i}")).collect::<Vec<_>>().join(",");
    let sql = format!(
        "SELECT c.id, c.session_id, s.source, coalesce(s.title,''), coalesce(s.repo,''), c.ts, c.kind,
                coalesce(c.text,''), c.event_first, c.hash
         FROM chunks c JOIN sessions s ON s.id = c.session_id
         WHERE c.hash IN ({marks}) AND c.kind = 'dialog'
           AND (?{a} IS NULL OR s.source = ?{a})
           AND (?{b} IS NULL OR instr(s.repo, ?{b}) > 0 OR instr(s.cwd, ?{b}) > 0)
           AND c.ts < ?{c} AND c.ts >= ?{d}",
        a = n + 1,
        b = n + 2,
        c = n + 3,
        d = n + 4,
    );
    use rusqlite::types::Value as V;
    let until = if q.include_recent { f64::MAX } else { now() - cfg.search.recent_secs };
    let mut values: Vec<V> = top.iter().map(|(i, _)| V::Text(vecs.hashes[*i].clone())).collect();
    values.push(q.source.clone().map_or(V::Null, V::Text));
    values.push(q.repo.clone().map_or(V::Null, V::Text));
    values.push(V::Real(until));
    values.push(V::Real(q.since.unwrap_or(0.0)));
    let mut st = store.conn.prepare(&sql)?;
    let mut hits: Vec<(usize, Hit)> = st
        .query_map(rusqlite::params_from_iter(values.iter()), |r| {
            let hash: String = r.get(9)?;
            let (rk, sim) = rank.get(hash.as_str()).copied().unwrap_or((usize::MAX, 0.0));
            Ok((
                rk,
                Hit {
                    chunk: r.get(0)?,
                    session: r.get(1)?,
                    source: r.get(2)?,
                    title: r.get(3)?,
                    repo: r.get(4)?,
                    ts: r.get(5)?,
                    kind: r.get(6)?,
                    event: r.get(8)?,
                    score: sim as f64,
                    text: r.get(7)?,
                    snippet: String::new(),
                },
            ))
        })?
        .collect::<Result<_, _>>()?;
    hits.sort_by_key(|(rk, h)| (*rk, h.chunk));
    Ok(hits.into_iter().map(|(_, h)| h).collect())
}

fn keyword_lane(store: &Store, cfg: &Config, q: &Query) -> anyhow::Result<Vec<Hit>> {
    let expr = match &q.raw_match {
        Some(m) => m.clone(),
        None => fts_query(&q.text, q.ko),
    };
    if expr.is_empty() {
        return Ok(vec![]);
    }
    let s = &cfg.search;
    let expr = if s.title_weight > 0.0 { expr } else { format!("body : ({expr})") };
    let fetch = (q.k * q.per_session.max(1) * 6).max(60);
    let lanes: Vec<Option<String>> = match (s.fusion, &q.kind) {
        (Fusion::Rrf, None) => vec![Some("dialog".into()), Some("tool".into())],
        _ => vec![q.kind.clone()],
    };
    let mut ranked: Vec<Hit> = Vec::new();
    if lanes.len() == 1 {
        ranked = fetch_lane(store, cfg, q, &expr, lanes[0].as_deref(), fetch)?;
    } else {
        // Reciprocal rank fusion of per-kind lanes, so long tool windows cannot crowd out dialog.
        let mut fused: HashMap<i64, (f64, Hit)> = HashMap::new();
        for (lane, weight) in lanes.iter().zip([s.dialog_weight, s.tool_weight]) {
            for (rank, hit) in fetch_lane(store, cfg, q, &expr, lane.as_deref(), fetch)?.into_iter().enumerate() {
                let add = weight / (s.rrf_k + rank as f64 + 1.0);
                fused.entry(hit.chunk).or_insert((0.0, hit)).0 += add;
            }
        }
        let mut v: Vec<(f64, Hit)> = fused.into_values().collect();
        v.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (score, mut hit) in v {
            hit.score = score * 100.0;
            ranked.push(hit);
        }
    }
    Ok(ranked)
}

fn fetch_lane(store: &Store, cfg: &Config, q: &Query, expr: &str, kind: Option<&str>, n: usize) -> anyhow::Result<Vec<Hit>> {
    let until = if q.include_recent { f64::MAX } else { now() - cfg.search.recent_secs };
    let mut st = store.conn.prepare_cached(
        "SELECT c.id, c.session_id, s.source, coalesce(s.title,''), coalesce(s.repo,''), c.ts, c.kind,
                c.text, c.span_start, c.span_len, c.event_first, e.tool, e.text, e.output,
                bm25(chunks_fts, 1.0, ?8)
         FROM chunks_fts
         JOIN chunks c ON c.id = chunks_fts.rowid
         JOIN sessions s ON s.id = c.session_id
         LEFT JOIN events e ON c.kind = 'tool' AND e.id = c.event_first
         WHERE chunks_fts MATCH ?1
           AND (?2 IS NULL OR s.source = ?2)
           AND (?3 IS NULL OR instr(s.repo, ?3) > 0 OR instr(s.cwd, ?3) > 0)
           AND (?4 IS NULL OR c.kind = ?4)
           AND c.ts < ?5 AND c.ts >= ?6
         ORDER BY bm25(chunks_fts, 1.0, ?8) LIMIT ?7",
    )?;
    let rows = st.query_map(
        params![expr, q.source, q.repo, kind, until, q.since.unwrap_or(0.0), n as i64, cfg.search.title_weight],
        |r| {
            let kind: String = r.get(6)?;
            let text = if kind == "tool" {
                let full = tool_text(
                    &r.get::<_, Option<String>>(11)?.unwrap_or_default(),
                    &r.get::<_, Option<String>>(12)?.unwrap_or_default(),
                    &r.get::<_, Option<String>>(13)?.unwrap_or_default(),
                );
                let (a, n) = (r.get::<_, Option<i64>>(8)?.unwrap_or(0), r.get::<_, Option<i64>>(9)?.unwrap_or(0));
                char_slice(&full, a as usize, n as usize).to_string()
            } else {
                r.get::<_, Option<String>>(7)?.unwrap_or_default()
            };
            Ok(Hit {
                chunk: r.get(0)?,
                session: r.get(1)?,
                source: r.get(2)?,
                title: r.get(3)?,
                repo: r.get(4)?,
                ts: r.get(5)?,
                kind,
                event: r.get(10)?,
                score: -r.get::<_, f64>(14)?,
                text,
                snippet: String::new(),
            })
        },
    )?;
    Ok(rows.collect::<Result<Vec<_>, _>>()?)
}

/// Dedupe, group and cut snippets.
fn finish(store: &Store, cfg: &Config, q: &Query, ranked: Vec<Hit>) -> anyhow::Result<Results> {
    let terms: Vec<String> = match &q.raw_match {
        Some(_) => words(&q.text),
        None => query_terms(&q.text, q.ko).into_iter().map(|(t, _)| t).collect(),
    };
    let excluded: HashSet<i64> = q.exclude_ids.iter().copied().collect();
    let mut seen_text = HashSet::new();
    let mut seen_event = HashSet::new();
    let mut order: Vec<String> = Vec::new();
    let mut groups: HashMap<String, Vec<Hit>> = HashMap::new();
    let mut matched = 0;
    for mut hit in ranked {
        if excluded.contains(&hit.chunk) {
            continue;
        }
        // One window per tool call, and repeated text (the same file read twice) once.
        if (hit.kind == "tool" && !seen_event.insert(hit.event)) || !seen_text.insert(hash(&hit.text)) {
            continue;
        }
        matched += 1;
        let key = match q.group {
            Group::Session => hit.session.clone(),
            Group::Chunk => hit.chunk.to_string(),
        };
        let g = groups.entry(key.clone()).or_default();
        if g.is_empty() {
            if order.len() >= q.k {
                continue;
            }
            order.push(key);
        }
        if g.len() < q.per_session.max(1) {
            hit.snippet = snippet(&hit.text, &terms, cfg.search.snippet);
            g.push(hit);
        }
    }
    let started: HashMap<String, f64> = {
        let ids: Vec<&String> = order.iter().filter_map(|k| groups.get(k).and_then(|g| g.first()).map(|h| &h.session)).collect();
        let mut m = HashMap::new();
        for id in ids {
            if let Some(t) = store
                .conn
                .query_row("SELECT started_at FROM sessions WHERE id=?1", [id], |r| r.get::<_, f64>(0))
                .optional()?
            {
                m.insert(id.clone(), t);
            }
        }
        m
    };
    let sessions = order
        .into_iter()
        .filter_map(|k| groups.remove(&k))
        .map(|hits| {
            let h = &hits[0];
            SessionHits {
                session: h.session.clone(),
                source: h.source.clone(),
                title: h.title.clone(),
                repo: h.repo.clone(),
                started_at: started.get(&h.session).copied().unwrap_or(h.ts),
                hits,
            }
        })
        .collect();
    Ok(Results { query: q.text.clone(), mode: String::new(), note: None, matched, sessions })
}

fn hash(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

/// `size` characters of `text` around the first query term.
pub fn snippet(text: &str, terms: &[String], size: usize) -> String {
    let n = text.chars().count();
    if n <= size {
        return text.to_string();
    }
    let lower = text.to_lowercase();
    let pos = terms
        .iter()
        .filter_map(|t| lower.find(t.as_str()))
        .min()
        .map(|b| lower[..b].chars().count())
        .unwrap_or(0);
    let start = pos.saturating_sub(size / 4).min(n - size);
    let body = char_slice(text, start, size);
    format!("{}{}{}", if start > 0 { "…" } else { "" }, body, if start + size < n { "…" } else { "" })
}

// ---- open -------------------------------------------------------------------------------

pub struct OpenArgs {
    pub reference: String,
    pub before: usize,
    pub after: usize,
    pub max_chars_per_event: usize,
    pub offset: usize,
}

/// A chunk id, `e<event id>` or a session id, with surrounding events.
pub fn open(store: &Store, a: &OpenArgs) -> anyhow::Result<String> {
    let r = a.reference.trim();
    let (session, first, last): (String, i64, i64) = if let Ok(chunk) = r.parse::<i64>() {
        store
            .conn
            .query_row("SELECT session_id, event_first, event_last FROM chunks WHERE id=?1", [chunk], |x| {
                Ok((x.get(0)?, x.get(1)?, x.get(2)?))
            })
            .optional()?
            .ok_or_else(|| anyhow::anyhow!("no chunk {chunk}"))?
    } else if let Some(ev) = r.strip_prefix('e').and_then(|x| x.parse::<i64>().ok()) {
        let s: String = store
            .conn
            .query_row("SELECT session_id FROM events WHERE id=?1", [ev], |x| x.get(0))
            .optional()?
            .ok_or_else(|| anyhow::anyhow!("no event {ev}"))?;
        (s, ev, ev)
    } else {
        let first: i64 = store
            .conn
            .query_row("SELECT min(id) FROM events WHERE session_id=?1", [r], |x| x.get::<_, Option<i64>>(0))?
            .ok_or_else(|| anyhow::anyhow!("no session {r:?}"))?;
        (r.to_string(), first, first)
    };
    let (source, title, repo, started, total): (String, String, String, f64, i64) = store.conn.query_row(
        "SELECT source, coalesce(title,''), coalesce(repo,''), started_at, (SELECT count(*) FROM events WHERE session_id=s.id)
         FROM sessions s WHERE id=?1",
        [&session],
        |x| Ok((x.get(0)?, x.get(1)?, x.get(2)?, x.get(3)?, x.get(4)?)),
    )?;
    let mut out = format!(
        "session {session} · {source} · {title} · {repo} · started {} · {total} events\n",
        fmt_time(started, true)
    );
    let mut q = store.conn.prepare(
        "SELECT id, seq, ts, role, coalesce(tool,''), coalesce(text,''), coalesce(output,''), coalesce(source_ref,'')
         FROM events WHERE session_id=?1 AND id >= ?2 AND id <= ?3 ORDER BY id",
    )?;
    let lo = first - a.before as i64;
    let hi = last + a.after as i64;
    let rows = q.query_map(params![session, lo, hi], |x| {
        Ok((
            x.get::<_, i64>(0)?,
            x.get::<_, i64>(1)?,
            x.get::<_, f64>(2)?,
            x.get::<_, String>(3)?,
            x.get::<_, String>(4)?,
            x.get::<_, String>(5)?,
            x.get::<_, String>(6)?,
            x.get::<_, String>(7)?,
        ))
    })?;
    for row in rows {
        let (id, seq, ts, role, tool, text, output, sref) = row?;
        let body = if role == "tool" { tool_text(&tool, &text, &output) } else { text };
        let n = body.chars().count();
        let start = if id >= first && id <= last { a.offset.min(n) } else { 0 };
        let take = a.max_chars_per_event.min(n - start);
        let mut piece = char_slice(&body, start, take).to_string();
        if start + take < n {
            piece += &format!("\n…[{} more characters; open e{id} with offset {}]", n - start - take, start + take);
        }
        let marker = if id >= first && id <= last { "▶ " } else { "" };
        out += &format!("\n{marker}[e{id} #{seq}] {} {role}{}\n{piece}\n", fmt_time(ts, false), if tool.is_empty() { String::new() } else { format!(" {tool}") });
        if !sref.is_empty() && id >= first && id <= last {
            out += &format!("(source: {sref})\n");
        }
    }
    Ok(out)
}

// ---- recent / files / fix ----------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct SessionRow {
    pub id: String,
    pub source: String,
    pub title: String,
    pub repo: String,
    pub started_at: f64,
    pub updated_at: f64,
    pub events: i64,
}

pub fn recent(store: &Store, n: usize, source: Option<&str>, repo: Option<&str>) -> anyhow::Result<Vec<SessionRow>> {
    let mut q = store.conn.prepare(
        "SELECT id, source, coalesce(title,''), coalesce(repo,''), started_at, updated_at,
                (SELECT count(*) FROM events WHERE session_id = s.id)
         FROM sessions s
         WHERE (?1 IS NULL OR source = ?1) AND (?2 IS NULL OR instr(repo, ?2) > 0 OR instr(cwd, ?2) > 0)
         ORDER BY updated_at DESC LIMIT ?3",
    )?;
    let rows = q.query_map(params![source, repo, n as i64], |r| {
        Ok(SessionRow {
            id: r.get(0)?,
            source: r.get(1)?,
            title: r.get(2)?,
            repo: r.get(3)?,
            started_at: r.get(4)?,
            updated_at: r.get(5)?,
            events: r.get(6)?,
        })
    })?;
    Ok(rows.flatten().collect())
}

/// Sessions that mention a file path, newest first.
pub fn files(store: &Store, cfg: &Config, path: &str, n: usize) -> anyhow::Result<Results> {
    let toks = words(path);
    if toks.is_empty() {
        anyhow::bail!("the path has no searchable words");
    }
    let mut q = Query::new(path, cfg);
    q.raw_match = Some(format!("\"{}\"", toks.join(" ").replace('"', "\"\"")));
    q.k = n;
    q.group = Group::Session;
    q.per_session = 1;
    let mut res = search(store, cfg, &q)?;
    res.sessions.sort_by(|a, b| b.hits[0].ts.total_cmp(&a.hits[0].ts));
    Ok(res)
}

#[derive(Debug, Serialize)]
pub struct FixRow {
    pub session: String,
    pub title: String,
    pub when: f64,
    pub error_chunk: i64,
    /// Tool calls that followed the error in that session: (event, tool, input, exit/err line).
    pub after: Vec<(i64, String, String, String)>,
}

/// What ran after the same error before.
pub fn fix(store: &Store, cfg: &Config, error: &str, n: usize, steps: usize) -> anyhow::Result<Vec<FixRow>> {
    let mut q = Query::new(error, cfg);
    q.kind = Some("tool".into());
    q.k = n;
    q.per_session = 1;
    let res = search(store, cfg, &q)?;
    let mut out = Vec::new();
    let mut st = store.conn.prepare(
        "SELECT id, coalesce(tool,''), coalesce(text,''), coalesce(output,'') FROM events
         WHERE session_id=?1 AND id > ?2 AND role='tool' ORDER BY id LIMIT ?3",
    )?;
    for s in res.sessions {
        let h = &s.hits[0];
        let after = st
            .query_map(params![s.session, h.event, steps as i64], |r| {
                let output: String = r.get(3)?;
                let status = output
                    .lines()
                    .rev()
                    .find(|l| l.starts_with("[exit") || l.starts_with("[error]"))
                    .unwrap_or("")
                    .to_string();
                Ok((r.get(0)?, r.get(1)?, crate::common::head_chars(&r.get::<_, String>(2)?, 300).to_string(), status))
            })?
            .flatten()
            .collect();
        out.push(FixRow { session: s.session.clone(), title: s.title.clone(), when: h.ts, error_chunk: h.chunk, after });
    }
    Ok(out)
}

pub fn fmt_time(ts: f64, date: bool) -> String {
    let dt = chrono::DateTime::from_timestamp(ts as i64, 0).unwrap_or_default().with_timezone(&chrono::Local);
    dt.format(if date { "%Y-%m-%d %H:%M" } else { "%m-%d %H:%M" }).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_lookups_to_keyword() {
        assert!(is_lexical("OOMKilled deploy api"));
        assert!(is_lexical("지난번 src/auth.rs 바꾼 이유가 뭐였지?"));
        assert!(is_lexical("배포 실패 원인 분석"));
        assert!(is_lexical("PR #1031 preflight"));
        assert!(!is_lexical("지난번에 배포가 왜 실패했는지 원인을 알려줄래?"));
        assert!(!is_lexical("Why did the deployment fail last time, and what did we change?"));
        assert!(!is_lexical("What did we decide about the read-only replica at that time."));
        assert!(is_lexical("why does main.rs panic on start"));
    }

    #[test]
    fn snippet_centres_on_the_first_term() {
        let text = format!("{}배포 실패 원인은 OOM{}", "a".repeat(300), "b".repeat(300));
        let s = snippet(&text, &["배포".to_string()], 60);
        assert!(s.contains("배포 실패"), "{s}");
        assert!(s.starts_with('…') && s.ends_with('…'));
        assert_eq!(snippet("short", &[], 60), "short");
    }
}
