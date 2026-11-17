//! OpenCode: ~/.local/share/opencode/opencode*.db, read directly (no sqlite3 CLI needed).
//!
//! OpenCode 2 writes only session_v2 / session_message and copies older sessions there; the old
//! session / message / part tables stop growing. Read v2 first and the old tables for whatever v2
//! does not hold, which also covers databases that were never migrated.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::common::{content_text, has_table, is_self_tool, open_ro, repo_name, str_of, tool_input};
use crate::model::{Candidate, Event, Session};

#[derive(Clone)]
struct Meta {
    id: String,
    title: String,
    cwd: String,
}

#[derive(Clone, Copy)]
enum Layout {
    V2,
    V1,
}

pub fn scan(home: &Path, since: f64) -> anyhow::Result<Vec<Candidate>> {
    let mut dbs: Vec<PathBuf> = match std::fs::read_dir(home) {
        Ok(rd) => rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                let n = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                n.starts_with("opencode") && n.ends_with(".db")
            })
            .collect(),
        Err(_) => return Ok(vec![]),
    };
    dbs.sort();
    let mut out = Vec::new();
    for db in dbs {
        let stem = db.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let prefix = if stem == "opencode" { String::new() } else { format!("{stem}/") };
        let Ok(conn) = open_ro(&db) else { continue };
        let mut seen = HashSet::new();
        for (layout, sessions, items) in [(Layout::V2, "session_v2", "session_message"), (Layout::V1, "session", "part")] {
            if !has_table(&conn, sessions) || !has_table(&conn, items) {
                continue;
            }
            let counts: HashMap<String, String> = {
                let mut q = conn.prepare(&format!("SELECT session_id, count(*), max(time_updated) FROM {items} GROUP BY session_id"))?;
                q.query_map([], |r| Ok((r.get::<_, String>(0)?, format!("{}:{}", r.get::<_, i64>(1)?, r.get::<_, Option<i64>>(2)?.unwrap_or(0)))))?
                    .flatten()
                    .collect()
            };
            let mut q = conn.prepare(&format!("SELECT id, title, directory, time_updated FROM {sessions} WHERE time_updated >= ?1"))?;
            let rows = q.query_map([(since * 1000.0) as i64], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, Option<String>>(1)?, r.get::<_, Option<String>>(2)?, r.get::<_, i64>(3)?))
            })?;
            for row in rows.flatten() {
                let (sid, title, dir, updated) = row;
                let Some(count) = counts.get(&sid) else { continue };
                if !seen.insert(sid.clone()) {
                    continue;
                }
                let meta = Meta { id: sid.clone(), title: title.unwrap_or_default(), cwd: dir.unwrap_or_default() };
                let (db, prefix) = (db.clone(), prefix.clone());
                out.push(Candidate {
                    id: format!("opencode:{prefix}{sid}"),
                    signature: format!("{updated}:{count}"),
                    load: Box::new(move || match layout {
                        Layout::V2 => load_v2(&db, &meta, &prefix, since),
                        Layout::V1 => load_v1(&db, &meta, &prefix, since),
                    }),
                });
            }
        }
    }
    Ok(out)
}

fn tool_event(ts: f64, name: &str, state: &Value, source_ref: String) -> Event {
    let mut out = content_text(state.get("output").or_else(|| state.get("content")).unwrap_or(&Value::Null));
    if out.is_empty() {
        if let Some(err) = state.get("error").filter(|e| !e.is_null()) {
            let msg = err.get("message").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| content_text(err));
            out = format!("[error] {msg}");
        }
    }
    Event::tool(ts, name, tool_input(state.get("input").unwrap_or(&Value::Null)), out, source_ref)
}

fn session(db: &Path, meta: &Meta, prefix: &str, events: Vec<Event>) -> Option<Session> {
    if events.is_empty() {
        return None;
    }
    Some(Session {
        id: format!("opencode:{prefix}{}", meta.id),
        source: "opencode",
        title: meta.title.chars().take(200).collect(),
        cwd: meta.cwd.clone(),
        repo: repo_name(&meta.cwd, None),
        source_ref: format!("{}#session/{}", db.display(), meta.id),
        events,
    })
}

/// A v2 assistant message holds text, reasoning and tool items; a tool item carries its input and
/// output together. System, synthetic and idle messages are not conversation.
fn load_v2(db: &Path, meta: &Meta, prefix: &str, since: f64) -> anyhow::Result<Option<Session>> {
    let conn = open_ro(db)?;
    let mut q = conn.prepare(
        "SELECT id, type, time_created, data FROM session_message WHERE session_id=?1 AND time_created >= ?2 ORDER BY seq",
    )?;
    let rows = q.query_map(rusqlite::params![meta.id, (since * 1000.0) as i64], |r| {
        Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, i64>(2)?, r.get::<_, String>(3)?))
    })?;
    let mut events = Vec::new();
    for row in rows.flatten() {
        let (mid, kind, ms, raw) = row;
        let Ok(d) = serde_json::from_str::<Value>(&raw) else { continue };
        let ts = ms as f64 / 1000.0;
        let r = format!("{}#session_message/{mid}", db.display());
        if kind == "user" && !str_of(&d, "text").trim().is_empty() {
            events.push(Event::msg(ts, "user", str_of(&d, "text"), r));
        } else if kind == "assistant" {
            for item in d.get("content").and_then(Value::as_array).into_iter().flatten() {
                match str_of(item, "type") {
                    "text" if !str_of(item, "text").trim().is_empty() => {
                        events.push(Event::msg(ts, "assistant", str_of(item, "text"), r.clone()))
                    }
                    "tool" if !is_self_tool(str_of(item, "name")) => events.push(tool_event(
                        ts,
                        str_of(item, "name"),
                        item.get("state").unwrap_or(&Value::Null),
                        r.clone(),
                    )),
                    _ => {}
                }
            }
        }
    }
    Ok(session(db, meta, prefix, events))
}

fn load_v1(db: &Path, meta: &Meta, prefix: &str, since: f64) -> anyhow::Result<Option<Session>> {
    let conn = open_ro(db)?;
    let roles: HashMap<String, String> = {
        let mut q = conn.prepare("SELECT id, data FROM message WHERE session_id=?1")?;
        q.query_map([&meta.id], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?
            .flatten()
            .map(|(id, data)| {
                let role = serde_json::from_str::<Value>(&data).ok().map(|v| str_of(&v, "role").to_string()).unwrap_or_default();
                (id, if role.is_empty() { "assistant".into() } else { role })
            })
            .collect()
    };
    let mut q = conn.prepare(
        "SELECT id, message_id, time_created, data FROM part WHERE session_id=?1 AND time_created >= ?2 ORDER BY time_created, id",
    )?;
    let rows = q.query_map(rusqlite::params![meta.id, (since * 1000.0) as i64], |r| {
        Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, i64>(2)?, r.get::<_, String>(3)?))
    })?;
    let mut events = Vec::new();
    for row in rows.flatten() {
        let (pid, mid, ms, raw) = row;
        let Ok(d) = serde_json::from_str::<Value>(&raw) else { continue };
        let ts = ms as f64 / 1000.0;
        let r = format!("{}#part/{pid}", db.display());
        let synthetic = d.get("synthetic").and_then(Value::as_bool).unwrap_or(false);
        match str_of(&d, "type") {
            "text" if !synthetic && !str_of(&d, "text").trim().is_empty() => {
                let role = roles.get(&mid).map(String::as_str).unwrap_or("assistant");
                events.push(Event::msg(ts, role, str_of(&d, "text"), r));
            }
            "tool" if !is_self_tool(str_of(&d, "tool")) => {
                events.push(tool_event(ts, str_of(&d, "tool"), d.get("state").unwrap_or(&Value::Null), r))
            }
            _ => {}
        }
    }
    Ok(session(db, meta, prefix, events))
}
