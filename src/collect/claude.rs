//! Claude Code: ~/.claude/projects/**/*.jsonl, tool_use paired with tool_result by id.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::common::{content_text, is_self_tool, iso_ts, repo_name, str_of, tool_input};
use crate::model::{Candidate, Event, Session};

pub fn scan(root: &Path, since: f64) -> anyhow::Result<Vec<Candidate>> {
    if !root.exists() {
        return Ok(vec![]);
    }
    let mut out = Vec::new();
    for entry in walkdir::WalkDir::new(root).follow_links(false).into_iter().flatten() {
        let path = entry.path();
        if !entry.file_type().is_file() || path.extension().is_none_or(|e| e != "jsonl") {
            continue;
        }
        let Ok(meta) = entry.metadata() else { continue };
        let mtime = meta.modified().ok().and_then(|m| m.duration_since(std::time::UNIX_EPOCH).ok()).unwrap_or_default();
        if mtime.as_secs_f64() < since {
            continue;
        }
        let native = native_id(path, root);
        let path: PathBuf = path.to_path_buf();
        let id = format!("claude:{native}");
        out.push(Candidate {
            id,
            signature: format!("{}:{}", meta.len(), mtime.as_nanos()),
            load: Box::new(move || load(&path, &native, since)),
        });
    }
    Ok(out)
}

/// The session uuid; a subagent transcript under <project>/<session>/subagents/ keeps its path.
fn native_id(path: &Path, root: &Path) -> String {
    let rel = path.strip_prefix(root).unwrap_or(path).with_extension("");
    let parts: Vec<String> = rel.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
    if parts.len() > 1 { parts[1..].join("/") } else { parts.join("/") }
}

pub fn load(path: &Path, native_id: &str, since: f64) -> anyhow::Result<Option<Session>> {
    let mut events: Vec<Event> = Vec::new();
    let mut pending: HashMap<String, usize> = HashMap::new();
    let mut skipped: HashSet<String> = HashSet::new();
    let (mut title, mut cwd) = (String::new(), String::new());
    let reader = BufReader::new(File::open(path)?);
    for (n, line) in reader.split(b'\n').map_while(Result::ok).enumerate() {
        let Ok(rec) = serde_json::from_slice::<Value>(&line) else { continue };
        let typ = str_of(&rec, "type");
        match typ {
            "ai-title" => {
                if let Some(t) = rec.get("aiTitle").and_then(Value::as_str).filter(|t| !t.is_empty()) {
                    title = t.to_string();
                }
                continue;
            }
            "custom-title" | "summary" => {
                if let Some(t) = ["customTitle", "summary"].iter().find_map(|k| rec.get(*k).and_then(Value::as_str)) {
                    title = t.to_string();
                }
                continue;
            }
            "user" | "assistant" => {}
            _ => continue,
        }
        if cwd.is_empty() {
            cwd = str_of(&rec, "cwd").to_string();
        }
        let flag = |k: &str| rec.get(k).and_then(Value::as_bool).unwrap_or(false);
        let Some(ts) = iso_ts(str_of(&rec, "timestamp")) else { continue };
        if ts < since || flag("isMeta") || flag("isApiErrorMessage") {
            continue;
        }
        let source_ref = format!("{}:{}", path.display(), n + 1);
        let content = rec.get("message").and_then(|m| m.get("content")).cloned().unwrap_or(Value::Null);
        if let Value::String(s) = &content {
            if !s.trim().is_empty() {
                events.push(Event::msg(ts, typ, s.clone(), source_ref));
            }
            continue;
        }
        for b in content.as_array().into_iter().flatten() {
            match str_of(b, "type") {
                "text" => {
                    let t = str_of(b, "text");
                    if !t.trim().is_empty() {
                        events.push(Event::msg(ts, typ, t, source_ref.clone()));
                    }
                }
                "tool_use" => {
                    let name = str_of(b, "name");
                    let id = str_of(b, "id").to_string();
                    if is_self_tool(name) {
                        skipped.insert(id);
                        continue;
                    }
                    pending.insert(id, events.len());
                    events.push(Event::tool(ts, name, tool_input(b.get("input").unwrap_or(&Value::Null)), "", source_ref.clone()));
                }
                "tool_result" => {
                    let tid = str_of(b, "tool_use_id");
                    if skipped.contains(tid) {
                        continue;
                    }
                    let mut out = content_text(b.get("content").unwrap_or(&Value::Null));
                    if b.get("is_error").and_then(Value::as_bool).unwrap_or(false) {
                        out = format!("[error] {out}");
                    }
                    match pending.remove(tid) {
                        Some(i) => events[i].output = out,
                        None => events.push(Event::tool(ts, "tool_result", "", out, source_ref.clone())),
                    }
                }
                _ => {}
            }
        }
    }
    if events.is_empty() {
        return Ok(None);
    }
    Ok(Some(Session {
        id: format!("claude:{native_id}"),
        source: "claude",
        title: title.chars().take(200).collect(),
        repo: repo_name(&cwd, None),
        cwd,
        source_ref: path.display().to_string(),
        events,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A made-up transcript written at test time; no recorded data lives in the repository.
    fn synthetic_transcript() -> std::path::PathBuf {
        let lines = [
            r#"{"type":"user","cwd":"/work/demo-repo","timestamp":"2026-01-01T01:00:00.000Z","message":{"role":"user","content":"why did the deploy fail?"}}"#,
            r#"{"type":"assistant","cwd":"/work/demo-repo","timestamp":"2026-01-01T01:00:05.000Z","message":{"role":"assistant","content":[{"type":"thinking","thinking":""},{"type":"text","text":"Checking logs."},{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"kubectl logs deploy/api | tail","description":"logs"}}]}}"#,
            r#"{"type":"user","cwd":"/work/demo-repo","timestamp":"2026-01-01T01:00:09.000Z","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"Error: OOMKilled exit 137"}]}}"#,
            r#"{"type":"assistant","cwd":"/work/demo-repo","timestamp":"2026-01-01T01:00:12.000Z","message":{"role":"assistant","content":[{"type":"tool_use","id":"t2","name":"mcp__ilji__search","input":{"query":"x"}}]}}"#,
            r#"{"type":"user","cwd":"/work/demo-repo","timestamp":"2026-01-01T01:00:13.000Z","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t2","content":"self result"}]}}"#,
            r#"{"type":"assistant","cwd":"/work/demo-repo","timestamp":"2026-01-01T01:00:20.000Z","isApiErrorMessage":true,"message":{"role":"assistant","content":[{"type":"text","text":"API error"}]}}"#,
            r#"{"type":"assistant","cwd":"/work/demo-repo","timestamp":"2026-01-01T01:00:30.000Z","message":{"role":"assistant","content":[{"type":"text","text":"Out of memory; raised the limit."}]}}"#,
            r#"{"type":"ai-title","aiTitle":"Deploy failure","sessionId":"s1"}"#,
        ];
        let dir = std::env::temp_dir().join(format!("ilji-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("s1.jsonl");
        std::fs::write(&path, lines.join("\n")).unwrap();
        path
    }

    #[test]
    fn loads_a_transcript() {
        let path = synthetic_transcript();
        let s = load(&path, "s1", 0.0).unwrap().unwrap();
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
        assert_eq!(s.title, "Deploy failure");
        assert_eq!(s.repo, "demo-repo");
        let roles: Vec<&str> = s.events.iter().map(|e| e.role.as_str()).collect();
        assert_eq!(roles, vec!["user", "assistant", "tool", "assistant"]); // API error and self search dropped
        assert_eq!(s.events[2].text, "kubectl logs deploy/api | tail");
        assert!(s.events[2].output.contains("OOMKilled"));
    }
}
