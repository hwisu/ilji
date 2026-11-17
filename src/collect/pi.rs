//! pi (pi-coding-agent): <agent dir>/sessions/<cwd>/<time>_<uuid>.jsonl.
//!
//! The first line is the session header (id, cwd). Message entries hold user, assistant and
//! toolResult messages; an assistant toolCall pairs with a toolResult by id. A session_info entry
//! names the session. Branches live in the same file and are all indexed.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::common::{content_text, is_self_tool, iso_ts, repo_name, str_of, tool_input};
use crate::model::{Candidate, Event, Session};

pub fn sessions_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("PI_CODING_AGENT_SESSION_DIR").filter(|v| !v.is_empty()) {
        return PathBuf::from(d);
    }
    std::env::var_os("PI_CODING_AGENT_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| crate::config::home().join(".pi/agent"))
        .join("sessions")
}

pub fn scan(root: &Path, since: f64) -> anyhow::Result<Vec<Candidate>> {
    if !root.exists() {
        return Ok(vec![]);
    }
    let mut out = Vec::new();
    for entry in walkdir::WalkDir::new(root).max_depth(3).into_iter().flatten() {
        let path = entry.path();
        if !entry.file_type().is_file() || path.extension().is_none_or(|e| e != "jsonl") {
            continue;
        }
        let Ok(meta) = entry.metadata() else { continue };
        let mtime = meta.modified().ok().and_then(|m| m.duration_since(std::time::UNIX_EPOCH).ok()).unwrap_or_default();
        if mtime.as_secs_f64() < since {
            continue;
        }
        // <time>_<uuid>.jsonl; the header id wins when the file is read.
        let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let native = stem.rsplit('_').next().unwrap_or(&stem).to_string();
        let path = path.to_path_buf();
        out.push(Candidate {
            id: format!("pi:{native}"),
            signature: format!("{}:{}", meta.len(), mtime.as_nanos()),
            load: Box::new(move || load(&path, &native, since)),
        });
    }
    Ok(out)
}

pub fn load(path: &Path, native_id: &str, since: f64) -> anyhow::Result<Option<Session>> {
    let mut events: Vec<Event> = Vec::new();
    let mut calls: HashMap<String, usize> = HashMap::new();
    let (mut cwd, mut name, mut first_user) = (String::new(), String::new(), String::new());
    for (n, line) in BufReader::new(File::open(path)?).split(b'\n').map_while(Result::ok).enumerate() {
        let Ok(rec) = serde_json::from_slice::<Value>(&line) else { continue };
        match str_of(&rec, "type") {
            "session" => {
                cwd = str_of(&rec, "cwd").to_string();
                continue;
            }
            "session_info" => {
                if !str_of(&rec, "name").is_empty() {
                    name = str_of(&rec, "name").to_string();
                }
                continue;
            }
            "message" => {}
            _ => continue,
        }
        let Some(ts) = iso_ts(str_of(&rec, "timestamp")) else { continue };
        if ts < since {
            continue;
        }
        let source_ref = format!("{}:{}", path.display(), n + 1);
        let m = rec.get("message").cloned().unwrap_or(Value::Null);
        match str_of(&m, "role") {
            "user" => {
                let text = content_text(m.get("content").unwrap_or(&Value::Null));
                if !text.trim().is_empty() {
                    if first_user.is_empty() {
                        first_user = text.clone();
                    }
                    events.push(Event::msg(ts, "user", text, source_ref));
                }
            }
            "assistant" => {
                for b in m.get("content").and_then(Value::as_array).into_iter().flatten() {
                    match str_of(b, "type") {
                        "text" if !str_of(b, "text").trim().is_empty() => {
                            events.push(Event::msg(ts, "assistant", str_of(b, "text"), source_ref.clone()))
                        }
                        "toolCall" => {
                            let tool = str_of(b, "name");
                            if is_self_tool(tool) {
                                continue;
                            }
                            calls.insert(str_of(b, "id").to_string(), events.len());
                            events.push(Event::tool(ts, tool, tool_input(b.get("arguments").unwrap_or(&Value::Null)), "", source_ref.clone()));
                        }
                        _ => {}
                    }
                }
            }
            "toolResult" => {
                let mut out = content_text(m.get("content").unwrap_or(&Value::Null));
                if m.get("isError").and_then(Value::as_bool).unwrap_or(false) {
                    out = format!("[error] {out}");
                }
                match calls.get(str_of(&m, "toolCallId")) {
                    Some(&i) => events[i].output = out,
                    None if !is_self_tool(str_of(&m, "toolName")) => {
                        events.push(Event::tool(ts, str_of(&m, "toolName"), "", out, source_ref))
                    }
                    None => {}
                }
            }
            _ => {}
        }
    }
    if events.is_empty() {
        return Ok(None);
    }
    let title = if name.is_empty() { first_user } else { name };
    Ok(Some(Session {
        id: format!("pi:{native_id}"),
        source: "pi",
        title: title.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(200).collect(),
        repo: repo_name(&cwd, None),
        cwd,
        source_ref: path.display().to_string(),
        events,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_a_pi_session() {
        let lines = [
            r#"{"type":"session","version":3,"id":"u1","timestamp":"2026-01-01T00:00:00.000Z","cwd":"/work/demo"}"#,
            r#"{"type":"message","id":"a","parentId":null,"timestamp":"2026-01-01T00:00:01.000Z","message":{"role":"system","content":"prompt"}}"#,
            r#"{"type":"message","id":"b","parentId":"a","timestamp":"2026-01-01T00:00:02.000Z","message":{"role":"user","content":[{"type":"text","text":"run the tests"}]}}"#,
            r#"{"type":"message","id":"c","parentId":"b","timestamp":"2026-01-01T00:00:03.000Z","message":{"role":"assistant","content":[{"type":"thinking","thinking":"..."},{"type":"toolCall","id":"call_1","name":"bash","arguments":{"command":"cargo test"}}]}}"#,
            r#"{"type":"message","id":"d","parentId":"c","timestamp":"2026-01-01T00:00:04.000Z","message":{"role":"toolResult","toolCallId":"call_1","toolName":"bash","content":[{"type":"text","text":"1 failed"}],"isError":true}}"#,
            r#"{"type":"message","id":"e","parentId":"d","timestamp":"2026-01-01T00:00:05.000Z","message":{"role":"assistant","content":[{"type":"text","text":"One test fails."}]}}"#,
            r#"{"type":"session_info","id":"f","parentId":"e","timestamp":"2026-01-01T00:00:06.000Z","name":"Fix tests"}"#,
        ];
        let dir = std::env::temp_dir().join(format!("ilji-pi-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("2026-01-01T00-00-00-000Z_u1.jsonl");
        std::fs::write(&path, lines.join("\n")).unwrap();
        let s = load(&path, "u1", 0.0).unwrap().unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!((s.id.as_str(), s.title.as_str(), s.repo.as_str()), ("pi:u1", "Fix tests", "demo"));
        let roles: Vec<&str> = s.events.iter().map(|e| e.role.as_str()).collect();
        assert_eq!(roles, vec!["user", "tool", "assistant"]);
        assert_eq!((s.events[1].text.as_str(), s.events[1].output.as_str()), ("cargo test", "[error] 1 failed"));
    }
}
