//! Settings. Precedence, lowest first: built-in defaults, the config file, `ILJI_<SECTION>_<KEY>`
//! environment variables, then `--set section.key=value` and per-command flags.
//!
//! Every switch and every resolution knob lives here, so the core reads one struct.

use std::path::PathBuf;

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};

/// Bump when parsing, redaction or chunking changes, so every session is collected again.
pub const CODE_VERSION: &str = "2";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ToolOutput {
    /// Index tool input and output.
    Full,
    /// Index the tool name and input only; output is not stored.
    Input,
    /// Do not index tool calls at all.
    Off,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Group {
    /// One entry per session with its best snippets (smaller answers).
    #[default]
    Session,
    /// One entry per chunk.
    Chunk,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Fusion {
    /// One BM25 ranking over all chunks.
    Bm25,
    /// Dialog and tool chunks ranked separately, then merged by reciprocal rank.
    Rrf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Mode {
    /// BM25 only.
    Keyword,
    /// Embeddings only (dialog chunks).
    Semantic,
    /// Both, merged as `search.hybrid` says. Falls back to keyword when no embedding is reachable.
    Hybrid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Hybrid {
    /// Reciprocal rank fusion of the keyword and embedding rankings.
    Rrf,
    /// Tool chunks keep their keyword ranks; only the dialog positions are re-ranked by RRF.
    /// Embeddings cover dialog only, so this keeps tool-output search as strong as keyword search.
    Slots,
    /// RRF where a tool chunk's keyword term is scaled to what a chunk found by both lanes reaches.
    Balanced,
    /// Route by the query: identifiers or a short keyword list go to keyword search (no embedding
    /// call); a sentence-like question goes to rrf.
    Auto,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Precision {
    /// One byte per dimension with a per-vector scale (a quarter of f32).
    I8,
    F32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Tools {
    /// search, open, recent, files, fix as separate MCP tools.
    Full,
    /// One `ilji` tool with a `mode` argument: the smallest schema per turn.
    Compact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Encrypt {
    None,
    /// The whole index encrypted at rest (needs a build with the `encrypt` feature).
    AtRest,
    /// Experimental: homomorphically encrypted vector lane. Not implemented yet.
    HeVectors,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Sources {
    pub codex: bool,
    pub claude: bool,
    pub opencode: bool,
    pub pi: bool,
    /// Pull requests and Actions runs through the `gh` CLI. Off by default: the only source that
    /// uses the network.
    pub github: bool,
}

impl Default for Sources {
    fn default() -> Self {
        Sources { codex: true, claude: true, opencode: true, pi: true, github: false }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IndexCfg {
    /// Days of history to keep.
    pub days: f64,
    /// Characters per dialog chunk.
    pub dialog_chunk: usize,
    /// Characters per tool-output window, and the overlap between windows.
    pub tool_window: usize,
    pub tool_overlap: usize,
    /// Tool output beyond this many characters is cut before storing.
    pub max_output: usize,
    pub tool_output: ToolOutput,
    /// Mask secrets before storing. Turning this off keeps raw secrets in the index.
    pub redact: bool,
    /// Sessions whose cwd, repo or id contains one of these strings are not indexed.
    pub exclude: Vec<String>,
}

impl Default for IndexCfg {
    fn default() -> Self {
        IndexCfg {
            days: 90.0,
            dialog_chunk: 1024,
            tool_window: 8192,
            tool_overlap: 1024,
            max_output: 256 * 1024,
            tool_output: ToolOutput::Full,
            redact: true,
            exclude: vec![],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SearchCfg {
    /// Results per query (sessions or chunks, by `group`).
    pub k: usize,
    pub group: Group,
    /// Snippets shown per session when grouping by session.
    pub per_session: usize,
    /// Characters per snippet.
    pub snippet: usize,
    /// Upper bound for one response, in characters.
    pub max_chars: usize,
    /// Activity this recent is left out unless asked for, so a session never finds itself.
    pub recent_secs: f64,
    /// Strip Korean particles and endings from query words.
    pub korean_stem: bool,
    /// Match Korean query words as prefixes.
    pub korean_prefix: bool,
    /// BM25 weight of the session title next to the chunk text (body = 1). 0 searches text only.
    pub title_weight: f64,
    pub fusion: Fusion,
    /// RRF constant and per-lane weights (used when fusion = rrf).
    pub rrf_k: f64,
    pub dialog_weight: f64,
    pub tool_weight: f64,
    pub mode: Mode,
    pub hybrid: Hybrid,
    /// Weights of the keyword and embedding rankings in hybrid fusion.
    pub keyword_weight: f64,
    pub dense_weight: f64,
    /// Embedding candidates considered per query.
    pub dense_k: usize,
    /// Multiplier on a tool chunk's keyword term in hybrid rrf (the embedding lane never returns tool chunks).
    pub tool_boost: f64,
}

impl Default for SearchCfg {
    fn default() -> Self {
        SearchCfg {
            k: 8,
            group: Group::Session,
            per_session: 2,
            snippet: 500,
            max_chars: 12_000,
            recent_secs: 300.0,
            korean_stem: true,
            korean_prefix: true,
            title_weight: 0.3,
            fusion: Fusion::Bm25,
            rrf_k: 60.0,
            dialog_weight: 1.0,
            tool_weight: 1.0,
            mode: Mode::Hybrid,
            hybrid: Hybrid::Auto,
            keyword_weight: 1.0,
            dense_weight: 1.0,
            dense_k: 100,
            tool_boost: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EmbedCfg {
    /// Embed dialog chunks and use them in search. Off = keyword only, no embedding server needed.
    pub enabled: bool,
    /// Embed new chunks at the end of `index` and each `watch` pass.
    pub auto: bool,
    /// An OpenAI-compatible embeddings endpoint (llama-server, Ollama, LM Studio, ...).
    pub url: String,
    /// Model name sent to the endpoint; vectors are stored per model.
    pub model: String,
    /// Text put before every query and every passage (instruction-tuned models need one).
    pub query_prefix: String,
    pub passage_prefix: String,
    pub precision: Precision,
    /// Texts per request and characters per text.
    pub batch: usize,
    pub max_chars: usize,
    pub timeout_secs: u64,
    /// Leave out tool transcripts that Codex stores inside imported assistant messages
    /// ([external_agent_tool_call] blocks); they stay in the keyword index.
    pub skip_imported_tools: bool,
    /// A chunk with fewer characters left than this is not embedded.
    pub min_chars: usize,
}

impl Default for EmbedCfg {
    fn default() -> Self {
        EmbedCfg {
            enabled: true,
            auto: true,
            url: "http://127.0.0.1:8767/v1/embeddings".into(),
            model: "harrier-oss-v1-0.6b".into(),
            query_prefix:
                "Instruct: Given a question about earlier coding work, retrieve the conversation passages that answer it\nQuery: ".into(),
            passage_prefix: String::new(),
            precision: Precision::I8,
            batch: 32,
            max_chars: 2048,
            timeout_secs: 120,
            skip_imported_tools: false,
            min_chars: 20,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServeCfg {
    pub tools: Tools,
    /// Characters per event in `open`.
    pub open_event_chars: usize,
}

impl Default for ServeCfg {
    fn default() -> Self {
        ServeCfg { tools: Tools::Full, open_event_chars: 4000 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Runs {
    /// Every Actions run outside pull requests (deploys, schedules, pushes).
    All,
    /// Only failed ones.
    Failed,
    Off,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GithubCfg {
    /// "owner/name" entries; "auto" adds the GitHub origins of clones that sessions ran in.
    pub repos: Vec<String>,
    /// Seconds between GitHub passes of `watch` / `index` (`index --source github` always runs).
    pub interval_secs: u64,
    /// API calls one pass may make; the rest waits for the next pass.
    pub max_calls: usize,
    /// Pull request comments and reviews.
    pub comments: bool,
    /// Changed file list of a pull request (so `ilji files` finds pull requests).
    pub files: bool,
    /// Failed checks of a pull request's recent commits.
    pub checks: bool,
    /// Actions runs outside pull requests.
    pub runs: Runs,
    /// Fetch an excerpt of failed job logs.
    pub logs: bool,
    /// Characters kept per job log, and lines kept around each error line.
    pub log_chars: usize,
    pub log_context: usize,
    /// Failed job logs fetched per pull request or run.
    pub logs_per_pr: usize,
}

impl Default for GithubCfg {
    fn default() -> Self {
        GithubCfg {
            repos: vec!["auto".into()],
            interval_secs: 300,
            max_calls: 300,
            comments: true,
            files: true,
            checks: true,
            runs: Runs::Failed,
            logs: true,
            log_chars: 16384,
            log_context: 10,
            logs_per_pr: 5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EncryptCfg {
    pub mode: Encrypt,
}

impl Default for EncryptCfg {
    fn default() -> Self {
        EncryptCfg { mode: Encrypt::None }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub sources: Sources,
    pub index: IndexCfg,
    pub search: SearchCfg,
    pub embed: EmbedCfg,
    pub serve: ServeCfg,
    pub encrypt: EncryptCfg,
    pub github: GithubCfg,
}

impl Config {
    /// Defaults, then the file, then the environment, then `overrides` ("section.key=value").
    pub fn load(overrides: &[String]) -> anyhow::Result<Config> {
        let mut root = toml::Value::try_from(Config::default())?;
        let path = config_path();
        if path.exists() {
            let text = std::fs::read_to_string(&path)?;
            let file: toml::Value = toml::from_str(&text).with_context(|| format!("reading {}", path.display()))?;
            merge(&mut root, file);
        }
        for (k, v) in std::env::vars() {
            let Some(rest) = k.strip_prefix("ILJI_") else { continue };
            let Some((section, key)) = rest.to_lowercase().split_once('_').map(|(a, b)| (a.to_string(), b.to_string())) else {
                continue;
            };
            let key_path = format!("{section}.{key}");
            if lookup(&root, &key_path).is_some() {
                set_path(&mut root, &key_path, &v)?;
            }
        }
        for o in overrides {
            let (k, v) = o.split_once('=').with_context(|| format!("--set wants key=value, got {o:?}"))?;
            set_path(&mut root, k.trim(), v.trim())?;
        }
        let cfg: Config = root.try_into().context("invalid configuration")?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        let i = &self.index;
        if i.dialog_chunk < 64 || i.tool_window < 256 {
            bail!("index.dialog_chunk must be ≥ 64 and index.tool_window ≥ 256");
        }
        if i.tool_overlap >= i.tool_window {
            bail!("index.tool_overlap must be smaller than index.tool_window");
        }
        if self.github.max_calls < 5 || self.github.log_chars < 500 {
            bail!("github.max_calls must be ≥ 5 and github.log_chars ≥ 500");
        }
        if self.search.k == 0 || self.search.snippet < 40 {
            bail!("search.k must be ≥ 1 and search.snippet ≥ 40");
        }
        Ok(())
    }

    /// A fingerprint of the settings that change what is stored. A change re-collects every session.
    pub fn index_fingerprint(&self) -> String {
        let i = &self.index;
        let key = format!("{}|{}|{}|{}|{:?}|{}", i.dialog_chunk, i.tool_window, i.tool_overlap, i.max_output, i.tool_output, i.redact);
        let mut h: u64 = 0xcbf29ce484222325;
        for b in key.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        format!("{CODE_VERSION}.{h:016x}")
    }

    pub fn source_enabled(&self, source: &str) -> bool {
        match source {
            "codex" => self.sources.codex,
            "claude" => self.sources.claude,
            "opencode" => self.sources.opencode,
            "pi" => self.sources.pi,
            "github" => self.sources.github,
            _ => false,
        }
    }
}

pub fn merge(base: &mut toml::Value, over: toml::Value) {
    match (base, over) {
        (toml::Value::Table(b), toml::Value::Table(o)) => {
            for (k, v) in o {
                match b.get_mut(&k) {
                    Some(slot) => merge(slot, v),
                    None => {
                        b.insert(k, v);
                    }
                }
            }
        }
        (slot, v) => *slot = v,
    }
}

pub fn lookup<'a>(root: &'a toml::Value, path: &str) -> Option<&'a toml::Value> {
    path.split('.').try_fold(root, |v, k| v.get(k))
}

/// Set `section.key` from text, parsed as the type the default has.
pub fn set_path(root: &mut toml::Value, path: &str, raw: &str) -> anyhow::Result<()> {
    let Some((section, key)) = path.split_once('.') else { bail!("use section.key, e.g. search.k") };
    let defaults = toml::Value::try_from(Config::default())?;
    let Some(template) = lookup(&defaults, path) else { bail!("unknown setting {path:?}; see `ilji config list`") };
    let value = match template {
        toml::Value::Boolean(_) => toml::Value::Boolean(match raw.to_lowercase().as_str() {
            "true" | "on" | "yes" | "1" => true,
            "false" | "off" | "no" | "0" => false,
            _ => bail!("{path} wants on/off, got {raw:?}"),
        }),
        toml::Value::Integer(_) => toml::Value::Integer(raw.parse().with_context(|| format!("{path} wants an integer"))?),
        toml::Value::Float(_) => toml::Value::Float(raw.parse().with_context(|| format!("{path} wants a number"))?),
        toml::Value::Array(_) => {
            toml::Value::Array(raw.split(',').map(str::trim).filter(|s| !s.is_empty()).map(|s| toml::Value::String(s.into())).collect())
        }
        _ => toml::Value::String(raw.to_string()),
    };
    let table = root.as_table_mut().context("config root")?;
    let sec = table.entry(section).or_insert_with(|| toml::Value::Table(Default::default()));
    sec.as_table_mut().context("config section")?.insert(key.to_string(), value);
    Ok(())
}

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var_os(key).filter(|v| !v.is_empty()).map(PathBuf::from)
}

pub fn home() -> PathBuf {
    dirs::home_dir().unwrap_or_else(|| PathBuf::from("."))
}

pub fn config_path() -> PathBuf {
    env_path("ILJI_CONFIG")
        .unwrap_or_else(|| env_path("XDG_CONFIG_HOME").unwrap_or_else(|| home().join(".config")).join("ilji/config.toml"))
}

pub fn data_dir() -> PathBuf {
    env_path("ILJI_HOME").unwrap_or_else(|| home().join(".local/share/ilji"))
}

/// The index file: ILJI_DB, or index.db in the data directory.
pub fn db_path() -> PathBuf {
    env_path("ILJI_DB").unwrap_or_else(|| data_dir().join("index.db"))
}

pub fn codex_home() -> PathBuf {
    env_path("CODEX_HOME").unwrap_or_else(|| home().join(".codex"))
}

pub fn claude_projects() -> PathBuf {
    env_path("CLAUDE_CONFIG_DIR").unwrap_or_else(|| home().join(".claude")).join("projects")
}

pub fn opencode_home() -> PathBuf {
    env_path("OPENCODE_DATA").unwrap_or_else(|| env_path("XDG_DATA_HOME").unwrap_or_else(|| home().join(".local/share")).join("opencode"))
}

pub fn now() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_parses_by_default_type() {
        let mut root = toml::Value::try_from(Config::default()).unwrap();
        set_path(&mut root, "search.k", "3").unwrap();
        set_path(&mut root, "search.korean_prefix", "off").unwrap();
        set_path(&mut root, "index.tool_output", "input").unwrap();
        set_path(&mut root, "index.exclude", "secret-repo, /tmp").unwrap();
        set_path(&mut root, "index.days", "30").unwrap();
        let cfg: Config = root.try_into().unwrap();
        assert_eq!(cfg.search.k, 3);
        assert!(!cfg.search.korean_prefix);
        assert_eq!(cfg.index.tool_output, ToolOutput::Input);
        assert_eq!(cfg.index.exclude, vec!["secret-repo", "/tmp"]);
        assert_eq!(cfg.index.days, 30.0);
    }

    #[test]
    fn rejects_unknown_and_bad_values() {
        let mut root = toml::Value::try_from(Config::default()).unwrap();
        assert!(set_path(&mut root, "search.nope", "1").is_err());
        assert!(set_path(&mut root, "search.k", "many").is_err());
        set_path(&mut root, "search.group", "planet").unwrap();
        assert!(root.try_into::<Config>().is_err());
    }

    #[test]
    fn fingerprint_follows_index_settings_only() {
        let a = Config::default();
        let mut b = Config::default();
        b.search.k = 99;
        assert_eq!(a.index_fingerprint(), b.index_fingerprint());
        b.index.tool_output = ToolOutput::Off;
        assert_ne!(a.index_fingerprint(), b.index_fingerprint());
    }
}
