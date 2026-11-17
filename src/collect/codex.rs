//! Codex: the app DB (thread_history_N.sqlite + state_N.sqlite) first, rollout files as fallback.
//!
//! Rollout .jsonl files can be missing or zstd-compressed while the app DB still holds every
//! turn, so the DB is the primary source.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use rusqlite::types::Value as Sql;
use serde_json::Value;

use crate::common::{columns, content_text, is_self_tool, iso_ts, open_ro, repo_name, str_of, tool_input};
use crate::model::{Candidate, Event, Session};

const SKIP_ITEMS: &[&str] =
    &["reasoning", "imageGeneration", "imageView", "sleep", "contextCompaction", "subAgentActivity", "collabAgentToolCall"];

/// The highest-numbered `<stem>_<N>.sqlite` in `home`.
fn newest(home: &Path, stem: &str) -> Option<PathBuf> {
    let mut best: Option<(u32, PathBuf)> = None;
    for entry in std::fs::read_dir(home).ok()?.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(n) = name.strip_prefix(&format!("{stem}_")).and_then(|r| r.strip_suffix(".sqlite")) else { continue };
        if let Ok(n) = n.parse::<u32>() {
            if best.as_ref().is_none_or(|(b, _)| n > *b) {
                best = Some((n, entry.path()));
            }
        }
    }
    best.map(|(_, p)| p)
}

#[derive(Clone, Debug, Default)]
struct Thread {
    id: String,
    title: String,
    cwd: String,
    origin: String,
    rollout_path: String,
    updated: f64,
}

pub fn scan(home: &Path, since: f64) -> anyhow::Result<Vec<Candidate>> {
    let Some(state_db) = newest(home, "state") else { return Ok(vec![]) };
    let hist_db = newest(home, "thread_history");
    let st = open_ro(&state_db)?;
    let cols = columns(&st, "threads");
    let has = |c: &str| cols.iter().any(|x| x == c);
    let mut filter = String::from("updated_at >= ?1");
    if has("thread_source") {
        filter += " AND coalesce(thread_source,'') != 'guardian_review'";
    }
    if has("source") {
        filter += " AND coalesce(source,'') NOT LIKE '%guardian%'";
    }
    let pick = |c: &str| if has(c) { c.to_string() } else { "NULL".to_string() };
    let sql = format!(
        "SELECT id, {}, {}, {}, {}, {}, {}, {} FROM threads WHERE {filter}",
        pick("name"),
        pick("title"),
        pick("first_user_message"),
        pick("cwd"),
        pick("git_origin_url"),
        pick("rollout_path"),
        if has("updated_at_ms") { "coalesce(updated_at_ms, updated_at * 1000)" } else { "updated_at * 1000" },
    );
    let text = |v: Sql| match v {
        Sql::Text(s) => s,
        _ => String::new(),
    };
    let mut threads = Vec::new();
    {
        let mut q = st.prepare(&sql)?;
        let mut rows = q.query([since as i64])?;
        while let Some(r) = rows.next()? {
            let title = [text(r.get(1)?), text(r.get(2)?), text(r.get(3)?)].into_iter().find(|t| !t.is_empty()).unwrap_or_default();
            threads.push(Thread {
                id: r.get(0)?,
                title: title.chars().take(200).collect(),
                cwd: text(r.get(4)?),
                origin: text(r.get(5)?),
                rollout_path: text(r.get(6)?),
                updated: match r.get::<_, Sql>(7)? {
                    Sql::Integer(i) => i as f64,
                    Sql::Real(f) => f,
                    _ => 0.0,
                },
            });
        }
    }

    let mut sigs: HashMap<String, (i64, i64)> = HashMap::new();
    if let Some(h) = hist_db.as_ref() {
        let conn = open_ro(h)?;
        let mut q =
            conn.prepare("SELECT thread_id, count(*), max(updated_at_ordinal) FROM thread_items GROUP BY thread_id")?;
        for row in q.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?, r.get::<_, Option<i64>>(2)?)))? {
            let (id, n, last) = row?;
            sigs.insert(id, (n, last.unwrap_or(0)));
        }
    }

    let mut out = Vec::new();
    for t in threads {
        let id = format!("codex:{}", t.id);
        if let (Some(h), Some(&(n, last))) = (hist_db.clone(), sigs.get(&t.id)) {
            if n > 0 {
                let signature = format!("db:{n}:{last}:{}", t.updated as i64);
                out.push(Candidate { id, signature, load: Box::new(move || load_db(&t, &h, since)) });
                continue;
            }
        }
        let Some(path) = rollout(&t.rollout_path) else { continue };
        let meta = std::fs::metadata(&path)?;
        let mtime = meta.modified().ok().and_then(|m| m.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_nanos());
        let signature = format!("file:{}:{mtime}", meta.len());
        out.push(Candidate { id, signature, load: Box::new(move || load_rollout(&t, &path, since)) });
    }
    Ok(out)
}

fn session(t: &Thread, source_ref: String, events: Vec<Event>) -> Option<Session> {
    if events.is_empty() {
        return None;
    }
    Some(Session {
        id: format!("codex:{}", t.id),
        source: "codex",
        title: t.title.clone(),
        cwd: t.cwd.clone(),
        repo: repo_name(&t.cwd, Some(&t.origin)),
        source_ref,
        events,
    })
}

fn load_db(t: &Thread, hist_db: &Path, since: f64) -> anyhow::Result<Option<Session>> {
    let conn = open_ro(hist_db)?;
    let mut q = conn.prepare(
        "SELECT item_id, item_type, item_json, created_at_ms FROM thread_items
         WHERE thread_id=?1 AND created_at_ms >= ?2 ORDER BY rollout_ordinal",
    )?;
    let rows = q.query_map(rusqlite::params![t.id, (since * 1000.0) as i64], |r| {
        Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?, r.get::<_, i64>(3)?))
    })?;
    let mut events = Vec::new();
    for row in rows {
        let (item_id, item_type, raw, ms) = row?;
        if SKIP_ITEMS.contains(&item_type.as_str()) {
            continue;
        }
        let Ok(d) = serde_json::from_str::<Value>(&raw) else { continue };
        if let Some(mut e) = item_event(&d, ms as f64 / 1000.0) {
            e.source_ref = format!("{}#thread_items/{}/{item_id}", hist_db.display(), t.id);
            events.push(e);
        }
    }
    Ok(session(t, format!("{}#thread/{}", hist_db.display(), t.id), events))
}

/// Map one thread_items.item_json to an event (None = not indexed).
pub fn item_event(d: &Value, ts: f64) -> Option<Event> {
    let named = |a: &str, b: &str| format!("{}.{}", str_of(d, a), str_of(d, b));
    match str_of(d, "type") {
        "userMessage" => {
            let text = content_text(d.get("content").unwrap_or(&Value::Null));
            (!text.trim().is_empty()).then(|| Event::msg(ts, "user", text, ""))
        }
        "agentMessage" | "plan" => {
            let text = str_of(d, "text");
            (!text.trim().is_empty()).then(|| Event::msg(ts, "assistant", text, ""))
        }
        "commandExecution" => {
            let mut out = str_of(d, "aggregatedOutput").to_string();
            if let Some(code) = d.get("exitCode").and_then(Value::as_i64).filter(|c| *c != 0) {
                out = format!("{out}\n[exit {code}]");
            }
            Some(Event::tool(ts, "shell", str_of(d, "command"), out, ""))
        }
        "mcpToolCall" => {
            let name = named("server", "tool");
            if is_self_tool(&name) {
                return None;
            }
            let out = match d.get("result").filter(|r| !r.is_null()) {
                Some(r) => content_text(r),
                None => d.get("error").map(Value::to_string).unwrap_or_default(),
            };
            Some(Event::tool(ts, name, tool_input(d.get("arguments").unwrap_or(&Value::Null)), out, ""))
        }
        "dynamicToolCall" => {
            let name = named("namespace", "tool");
            if is_self_tool(&name) {
                return None;
            }
            let out = content_text(d.get("contentItems").unwrap_or(&Value::Null));
            Some(Event::tool(ts, name, tool_input(d.get("arguments").unwrap_or(&Value::Null)), out, ""))
        }
        "fileChange" => {
            let mut parts = Vec::new();
            for c in d.get("changes").and_then(Value::as_array).into_iter().flatten() {
                let kind = c.get("kind").map(|k| str_of(k, "type")).unwrap_or("");
                parts.push(format!("{kind} {}\n{}", str_of(c, "path"), str_of(c, "diff")).trim_end().to_string());
            }
            Some(Event::tool(ts, "apply_patch", parts.join("\n"), str_of(d, "status"), ""))
        }
        "webSearch" => {
            let action = d.get("action").cloned().unwrap_or(Value::Null);
            let query = [str_of(d, "query"), str_of(&action, "query"), str_of(&action, "url")]
                .into_iter()
                .find(|s| !s.is_empty())
                .unwrap_or("");
            let res: Vec<String> = d
                .get("results")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .map(|r| {
                    [str_of(r, "title"), str_of(r, "url"), str_of(r, "snippet")]
                        .into_iter()
                        .filter(|s| !s.is_empty())
                        .collect::<Vec<_>>()
                        .join(" | ")
                })
                .collect();
            Some(Event::tool(ts, "web_search", query, res.join("\n"), ""))
        }
        "functionCallOutput" => {
            let name = named("namespace", "name").trim_start_matches('.').to_string();
            if is_self_tool(&name) {
                return None;
            }
            Some(Event::tool(ts, name, "", content_text(d.get("output").unwrap_or(&Value::Null)), ""))
        }
        _ => None,
    }
}

// --- rollout fallback ---------------------------------------------------------------

fn rollout(path: &str) -> Option<PathBuf> {
    if path.is_empty() {
        return None;
    }
    [PathBuf::from(path), PathBuf::from(format!("{path}.zst"))].into_iter().find(|p| p.exists())
}

fn lines(path: &Path) -> anyhow::Result<Box<dyn Iterator<Item = String>>> {
    let file = File::open(path)?;
    let reader: Box<dyn Read> =
        if path.extension().is_some_and(|e| e == "zst") { Box::new(zstd::Decoder::new(file)?) } else { Box::new(file) };
    Ok(Box::new(BufReader::new(reader).lines().map_while(Result::ok)))
}

fn load_rollout(t: &Thread, path: &Path, since: f64) -> anyhow::Result<Option<Session>> {
    let mut events: Vec<Event> = Vec::new();
    let mut calls: HashMap<String, usize> = HashMap::new();
    for (n, line) in lines(path)?.enumerate() {
        let Ok(rec) = serde_json::from_str::<Value>(&line) else { continue };
        if str_of(&rec, "type") != "response_item" {
            continue;
        }
        let p = rec.get("payload").cloned().unwrap_or(Value::Null);
        let ts = iso_ts(str_of(&rec, "timestamp")).unwrap_or(t.updated / 1000.0);
        if ts < since {
            continue;
        }
        let source_ref = format!("{}:{}", path.display(), n + 1);
        match str_of(&p, "type") {
            "message" => {
                let role = str_of(&p, "role");
                if role != "user" && role != "assistant" {
                    continue;
                }
                let text = content_text(p.get("content").unwrap_or(&Value::Null));
                let head = text.trim_start();
                if !text.trim().is_empty()
                    && !["<environment_context>", "<user_instructions>", "# AGENTS.md"].iter().any(|m| head.starts_with(m))
                {
                    events.push(Event::msg(ts, role, text, source_ref));
                }
            }
            "function_call" | "custom_tool_call" | "local_shell_call" => {
                let name = Some(str_of(&p, "name")).filter(|s| !s.is_empty()).unwrap_or("shell").to_string();
                if is_self_tool(&name) {
                    continue;
                }
                let args = ["arguments", "input", "action"].iter().find_map(|k| p.get(*k).filter(|v| !v.is_null()));
                let e = Event::tool(ts, name, tool_input(args.unwrap_or(&Value::Null)), "", source_ref);
                calls.insert(str_of(&p, "call_id").to_string(), events.len());
                events.push(e);
            }
            "function_call_output" | "custom_tool_call_output" => {
                if let Some(&i) = calls.get(str_of(&p, "call_id")) {
                    let out = match p.get("output") {
                        Some(Value::String(s)) => serde_json::from_str::<Value>(s)
                            .ok()
                            .and_then(|v| v.get("output").and_then(Value::as_str).map(str::to_string))
                            .unwrap_or_else(|| s.clone()),
                        Some(v) => content_text(v),
                        None => String::new(),
                    };
                    events[i].output = out;
                }
            }
            _ => {}
        }
    }
    Ok(session(t, path.display().to_string(), events))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn maps_items() {
        let e = item_event(&json!({"type": "commandExecution", "command": "ls", "aggregatedOutput": "x", "exitCode": 2}), 1.0)
            .unwrap();
        assert_eq!((e.tool.as_str(), e.text.as_str(), e.output.as_str()), ("shell", "ls", "x\n[exit 2]"));
        let e = item_event(&json!({"type": "userMessage", "content": [{"type": "text", "text": "안녕"}]}), 1.0).unwrap();
        assert_eq!((e.role.as_str(), e.text.as_str()), ("user", "안녕"));
        assert!(item_event(&json!({"type": "mcpToolCall", "server": "ilji", "tool": "search"}), 1.0).is_none());
        assert!(item_event(&json!({"type": "reasoning"}), 1.0).is_none());
    }
}
