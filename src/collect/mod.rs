//! Source collectors. Each `scan(since)` lists candidates with a cheap change signature.

pub mod claude;
pub mod codex;
pub mod opencode;
pub mod pi;

use std::path::PathBuf;

use crate::model::Candidate;

pub const SOURCES: &[&str] = &["codex", "claude", "opencode", "pi"];

pub fn root(source: &str) -> PathBuf {
    match source {
        "codex" => crate::config::codex_home(),
        "claude" => crate::config::claude_projects(),
        "opencode" => crate::config::opencode_home(),
        _ => pi::sessions_dir(),
    }
}

pub fn scan(source: &str, since: f64) -> anyhow::Result<Vec<Candidate>> {
    let root = root(source);
    match source {
        "codex" => codex::scan(&root, since),
        "claude" => claude::scan(&root, since),
        "opencode" => opencode::scan(&root, since),
        "pi" => pi::scan(&root, since),
        other => anyhow::bail!("unknown source {other:?}; use one of {SOURCES:?}"),
    }
}
