//! Source collectors. Each `scan(since)` lists candidates with a cheap change signature.

pub mod claude;
pub mod codex;
pub mod github;
pub mod opencode;
pub mod pi;

use std::path::PathBuf;

use crate::model::Candidate;

pub const SOURCES: &[&str] = &["codex", "claude", "opencode", "pi", "github"];

/// What a scan may use besides the source's own files.
pub struct ScanCtx<'a> {
    pub since: f64,
    pub cfg: &'a crate::config::Config,
    /// The source was asked for by name (`index --source github`), so throttles do not apply.
    pub explicit: bool,
    /// Whether the index holds this session with this signature already.
    pub known: &'a (dyn Fn(&str, &str) -> bool + Sync),
}

pub fn root(source: &str) -> PathBuf {
    match source {
        "codex" => crate::config::codex_home(),
        "claude" => crate::config::claude_projects(),
        "opencode" => crate::config::opencode_home(),
        "github" => crate::config::data_dir().join("github-state.json"),
        _ => pi::sessions_dir(),
    }
}

pub fn scan(source: &str, ctx: &ScanCtx) -> anyhow::Result<Vec<Candidate>> {
    let root = root(source);
    let since = ctx.since;
    match source {
        "codex" => codex::scan(&root, since),
        "claude" => claude::scan(&root, since),
        "opencode" => opencode::scan(&root, since),
        "pi" => pi::scan(&root, since),
        "github" => github::scan(ctx),
        other => anyhow::bail!("unknown source {other:?}; use one of {SOURCES:?}"),
    }
}
