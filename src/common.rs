//! Helpers shared by the collectors.

use std::path::Path;
use std::sync::LazyLock;

use regex::Regex;
use rusqlite::{Connection, OpenFlags};
use serde_json::Value;

const SELF_TOOL_PREFIXES: &[&str] =
    &["mcp__ilji__", "ilji_", "mcp__agent-history__", "mcp__agent_history__", "agent-history_", "agent_history_"];
const SELF_SERVERS: &[&str] = &["ilji", "agent-history", "agent_history"];

/// Search output, however it was produced (MCP, the CLI in a shell, an eval script).
/// Indexing it would let the index quote itself.
static SELF_OUTPUT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^(?:\d+ hits for |### chunk \d+ · |session \S+ · .* · started \d{4}-)").unwrap());

pub fn is_self_tool(name: &str) -> bool {
    SELF_TOOL_PREFIXES.iter().any(|p| name.starts_with(p)) || SELF_SERVERS.contains(&name.split('.').next().unwrap_or(""))
}

pub fn is_self_output(text: &str) -> bool {
    !text.is_empty() && SELF_OUTPUT_RE.is_match(head_chars(text, 20_000))
}

/// The first `n` characters of `s`, cut on a character boundary.
pub fn head_chars(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

pub fn repo_name(cwd: &str, origin: Option<&str>) -> String {
    if let Some(o) = origin.filter(|o| !o.is_empty()) {
        let tail = o.trim_end_matches('/').rsplit('/').next().unwrap_or(o);
        let tail = tail.rsplit(':').next().unwrap_or(tail);
        return tail.strip_suffix(".git").unwrap_or(tail).to_string();
    }
    Path::new(cwd.trim_end_matches('/')).file_name().and_then(|n| n.to_str()).unwrap_or("").to_string()
}

/// Readable tool input: the shell command when there is one, compact JSON otherwise.
pub fn tool_input(value: &Value) -> String {
    let parsed;
    let value = match value {
        Value::String(s) => match serde_json::from_str::<Value>(s) {
            Ok(v) => {
                parsed = v;
                &parsed
            }
            Err(_) => return s.clone(),
        },
        v => v,
    };
    if let Value::Object(map) = value {
        let cmd = match map.get("command").or_else(|| map.get("cmd")) {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(parts)) => {
                parts.iter().map(|p| p.as_str().map(str::to_string).unwrap_or_else(|| p.to_string())).collect::<Vec<_>>().join(" ")
            }
            _ => String::new(),
        };
        if !cmd.is_empty() {
            let rest: serde_json::Map<String, Value> = map
                .iter()
                .filter(|(k, _)| !matches!(k.as_str(), "command" | "cmd" | "description"))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            return if rest.is_empty() { cmd } else { format!("{cmd}\n{}", Value::Object(rest)) };
        }
    }
    match value {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        // key: value lines, strings unescaped, so file contents and patches read (and match) as text.
        Value::Object(m) => m
            .iter()
            .filter(|(_, v)| !v.is_null())
            .map(|(k, v)| match v {
                Value::String(s) => format!("{k}: {s}"),
                v => format!("{k}: {v}"),
            })
            .collect::<Vec<_>>()
            .join("\n"),
        v => v.to_string(),
    }
}

/// Flatten MCP/Anthropic-style content (a string or a list of blocks) to text.
pub fn content_text(content: &Value) -> String {
    match content {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Object(m) => match m.get("content") {
            Some(c) => content_text(c),
            None => m.get("text").and_then(Value::as_str).unwrap_or("").to_string(),
        },
        Value::Array(blocks) => {
            let mut parts = Vec::new();
            for b in blocks {
                match b {
                    Value::String(s) => parts.push(s.clone()),
                    Value::Object(m) => {
                        let t = m.get("type").and_then(Value::as_str).unwrap_or("");
                        if matches!(t, "text" | "input_text" | "output_text" | "inputText") {
                            parts.push(m.get("text").and_then(Value::as_str).unwrap_or("").to_string());
                        }
                    }
                    _ => {}
                }
            }
            parts.retain(|p| !p.is_empty());
            parts.join("\n")
        }
        v => v.to_string(),
    }
}

pub fn str_of<'a>(v: &'a Value, key: &str) -> &'a str {
    v.get(key).and_then(Value::as_str).unwrap_or("")
}

/// RFC 3339 timestamp to Unix seconds.
pub fn iso_ts(s: &str) -> Option<f64> {
    chrono::DateTime::parse_from_rfc3339(s).ok().map(|d| d.timestamp_millis() as f64 / 1000.0)
}

/// Open a source database read-only. Collectors never write to source stores.
pub fn open_ro(path: &Path) -> rusqlite::Result<Connection> {
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX)?;
    conn.busy_timeout(std::time::Duration::from_secs(10))?;
    Ok(conn)
}

pub fn has_table(conn: &Connection, name: &str) -> bool {
    conn.query_row("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1", [name], |_| Ok(())).is_ok()
}

pub fn columns(conn: &Connection, table: &str) -> Vec<String> {
    let Ok(mut st) = conn.prepare(&format!("PRAGMA table_info({table})")) else { return vec![] };
    st.query_map([], |r| r.get::<_, String>(1)).map(|rows| rows.flatten().collect()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tool_input_prefers_the_command() {
        assert_eq!(tool_input(&json!({"command": "ls -la", "description": "list"})), "ls -la");
        assert_eq!(tool_input(&json!({"cmd": ["git", "status"], "workdir": "/x"})), "git status\n{\"workdir\":\"/x\"}");
        assert_eq!(tool_input(&json!("{\"command\":\"pwd\"}")), "pwd");
        assert_eq!(tool_input(&json!("plain text")), "plain text");
        assert_eq!(tool_input(&json!({})), "");
        assert_eq!(tool_input(&json!({"file_path": "a.rs", "content": "fn x() {\n}"})), "content: fn x() {\n}\nfile_path: a.rs");
    }

    #[test]
    fn content_text_flattens_blocks() {
        let v = json!([{"type": "text", "text": "a"}, {"type": "image"}, "b", {"type": "output_text", "text": "c"}]);
        assert_eq!(content_text(&v), "a\nb\nc");
        assert_eq!(content_text(&json!({"content": [{"type": "text", "text": "x"}]})), "x");
    }

    #[test]
    fn repo_from_origin_or_cwd() {
        assert_eq!(repo_name("/home/u/work/api", None), "api");
        assert_eq!(repo_name("/x", Some("git@github.com:org/web.git")), "web");
        assert_eq!(repo_name("/x", Some("https://github.com/org/web/")), "web");
    }

    #[test]
    fn self_detection() {
        assert!(is_self_tool("mcp__ilji__search"));
        assert!(is_self_tool("agent-history.history_search"));
        assert!(!is_self_tool("Bash"));
        assert!(is_self_output("3 hits for 'x'\n\n### chunk 12 · codex · ..."));
        assert!(!is_self_output("ordinary output"));
    }
}
