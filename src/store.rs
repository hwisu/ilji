//! The index: sessions, events, chunks and a contentless FTS5 table over chunk text.
//! Tool chunks are spans into their event, so long outputs are stored once.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Context;
use rusqlite::{Connection, OpenFlags, OptionalExtension, params};

use crate::chunk::{Body, char_slice, chunk_events, tool_text};
use crate::config::IndexCfg;
use crate::model::Session;

pub const SCHEMA_VERSION: &str = "3";

/// FNV-1a 64-bit: stable across builds, used to key vectors by chunk text.
pub fn text_hash(s: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

const SCHEMA: &str = r#"
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS sessions(
  id TEXT PRIMARY KEY, source TEXT NOT NULL, title TEXT, cwd TEXT, repo TEXT,
  started_at REAL, updated_at REAL, signature TEXT, source_ref TEXT);
CREATE INDEX IF NOT EXISTS sessions_updated ON sessions(updated_at);
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, seq INTEGER, ts REAL, role TEXT,
  tool TEXT, text TEXT, output TEXT, source_ref TEXT);
CREATE INDEX IF NOT EXISTS events_session ON events(session_id, seq);
CREATE TABLE IF NOT EXISTS chunks(
  id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, event_first INTEGER, event_last INTEGER,
  ts REAL, kind TEXT, text TEXT, span_start INTEGER, span_len INTEGER, hash TEXT);
CREATE INDEX IF NOT EXISTS chunks_session ON chunks(session_id);
CREATE INDEX IF NOT EXISTS chunks_hash ON chunks(hash);
-- Embeddings keyed by chunk text, so a re-read session with unchanged text is not embedded again.
CREATE TABLE IF NOT EXISTS vectors(
  model TEXT NOT NULL, hash TEXT NOT NULL, dim INTEGER NOT NULL, scale REAL NOT NULL, data BLOB NOT NULL,
  PRIMARY KEY(model, hash)) WITHOUT ROWID;
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  body, title, content='', contentless_delete=1, tokenize='unicode61 remove_diacritics 2');
CREATE TABLE IF NOT EXISTS forgotten(id TEXT PRIMARY KEY, at REAL);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
"#;

pub struct Store {
    pub conn: Connection,
}

#[derive(Debug, Default)]
pub struct Stats {
    pub sources: Vec<(String, i64, i64, i64)>, // source, sessions, events, chunks
    pub first: f64,
    pub last: f64,
    pub bytes: u64,
    pub forgotten: i64,
    /// (model, vectors, dialog chunks with a vector for that model)
    pub vectors: Vec<(String, i64, i64)>,
    pub dialog_chunks: i64,
}

impl Store {
    pub fn open(path: &Path) -> anyhow::Result<Store> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700));
            }
        }
        let conn = Connection::open(path).with_context(|| format!("opening {}", path.display()))?;
        conn.busy_timeout(std::time::Duration::from_secs(30))?;
        // The index is a cache of the source stores: on a schema change it is rebuilt, not migrated.
        let version: Option<String> =
            conn.query_row("SELECT value FROM meta WHERE key='schema'", [], |r| r.get(0)).optional().unwrap_or(None);
        if version.as_deref() != Some(SCHEMA_VERSION) {
            conn.execute_batch(
                "DROP TABLE IF EXISTS chunks_fts; DROP TABLE IF EXISTS chunks; DROP TABLE IF EXISTS events;
                 DROP TABLE IF EXISTS sessions;",
            )?;
        }
        conn.execute_batch(SCHEMA)?;
        conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('schema', ?1)", [SCHEMA_VERSION])?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
        }
        Ok(Store { conn })
    }

    /// Read-only, for search and the MCP server. Fails if the index does not exist yet.
    pub fn open_ro(path: &Path) -> anyhow::Result<Store> {
        if !path.exists() {
            anyhow::bail!("no index at {}; run `ilji index` first", path.display());
        }
        let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX)?;
        conn.busy_timeout(std::time::Duration::from_secs(10))?;
        Ok(Store { conn })
    }

    pub fn signatures(&self) -> anyhow::Result<HashMap<String, String>> {
        let mut q = self.conn.prepare("SELECT id, signature FROM sessions")?;
        let rows = q.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, Option<String>>(1)?.unwrap_or_default())))?;
        Ok(rows.flatten().collect())
    }

    pub fn is_forgotten(&self, id: &str) -> bool {
        self.conn.query_row("SELECT 1 FROM forgotten WHERE id=?1", [id], |_| Ok(())).is_ok()
    }

    fn delete_rows(tx: &Connection, id: &str) -> rusqlite::Result<()> {
        tx.execute("DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE session_id=?1)", [id])?;
        tx.execute("DELETE FROM chunks WHERE session_id=?1", [id])?;
        tx.execute("DELETE FROM events WHERE session_id=?1", [id])?;
        tx.execute("DELETE FROM sessions WHERE id=?1", [id])?;
        Ok(())
    }

    pub fn delete_session(&mut self, id: &str) -> anyhow::Result<bool> {
        let tx = self.conn.transaction()?;
        let existed = tx.query_row("SELECT 1 FROM sessions WHERE id=?1", [id], |_| Ok(())).optional()?.is_some();
        Self::delete_rows(&tx, id)?;
        tx.commit()?;
        Ok(existed)
    }

    /// Drop a session and keep it out of later ingests.
    pub fn forget(&mut self, id: &str) -> anyhow::Result<bool> {
        let existed = self.delete_session(id)?;
        self.conn.execute("INSERT OR REPLACE INTO forgotten(id, at) VALUES(?1, ?2)", params![id, crate::config::now()])?;
        Ok(existed)
    }

    pub fn unforget(&mut self, id: &str) -> anyhow::Result<bool> {
        Ok(self.conn.execute("DELETE FROM forgotten WHERE id=?1", [id])? > 0)
    }

    /// Write a session, replacing what was stored for it. Returns the number of chunks.
    pub fn replace_session(&mut self, s: &Session, signature: &str, cfg: &IndexCfg) -> anyhow::Result<usize> {
        let tx = self.conn.transaction()?;
        Self::delete_rows(&tx, &s.id)?;
        tx.execute(
            "INSERT INTO sessions(id, source, title, cwd, repo, started_at, updated_at, signature, source_ref)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![s.id, s.source, s.title, s.cwd, s.repo, s.started_at(), s.updated_at(), signature, s.source_ref],
        )?;
        let mut event_ids = Vec::with_capacity(s.events.len());
        {
            let mut ins = tx.prepare_cached(
                "INSERT INTO events(session_id, seq, ts, role, tool, text, output, source_ref)
                 VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            )?;
            for (i, e) in s.events.iter().enumerate() {
                ins.execute(params![s.id, i as i64, e.ts, e.role, e.tool, e.text, e.output, e.source_ref])?;
                event_ids.push(tx.last_insert_rowid());
            }
        }
        let chunks = chunk_events(&s.events, cfg);
        {
            let mut ins_chunk = tx.prepare_cached(
                "INSERT INTO chunks(session_id, event_first, event_last, ts, kind, text, span_start, span_len, hash)
                 VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            )?;
            let mut ins_fts = tx.prepare_cached("INSERT INTO chunks_fts(rowid, body, title) VALUES(?1, ?2, ?3)")?;
            let mut cached: Option<(usize, String)> = None;
            for c in &chunks {
                let (first, last) = (event_ids[c.first], event_ids[c.last]);
                match &c.body {
                    Body::Dialog(text) => {
                        ins_chunk.execute(params![s.id, first, last, c.ts, "dialog", text, None::<i64>, None::<i64>, text_hash(text)])?;
                        ins_fts.execute(params![tx.last_insert_rowid(), text, s.title])?;
                    }
                    Body::Tool { start, len } => {
                        if cached.as_ref().is_none_or(|(i, _)| *i != c.first) {
                            let e = &s.events[c.first];
                            cached = Some((c.first, tool_text(&e.tool, &e.text, &e.output)));
                        }
                        let body = char_slice(&cached.as_ref().unwrap().1, *start, *len);
                        ins_chunk.execute(params![
                            s.id,
                            first,
                            last,
                            c.ts,
                            "tool",
                            None::<String>,
                            *start as i64,
                            *len as i64,
                            None::<String>
                        ])?;
                        ins_fts.execute(params![tx.last_insert_rowid(), body, s.title])?;
                    }
                }
            }
        }
        tx.commit()?;
        Ok(chunks.len())
    }

    /// Delete sessions last updated before `before`. Returns how many went.
    pub fn prune(&mut self, before: f64) -> anyhow::Result<usize> {
        let ids: Vec<String> = {
            let mut q = self.conn.prepare("SELECT id FROM sessions WHERE updated_at < ?1")?;
            q.query_map([before], |r| r.get(0))?.flatten().collect()
        };
        for id in &ids {
            self.delete_session(id)?;
        }
        Ok(ids.len())
    }

    /// Drop vectors no chunk refers to any more. Returns how many went.
    pub fn prune_vectors(&self) -> anyhow::Result<usize> {
        Ok(self.conn.execute("DELETE FROM vectors WHERE hash NOT IN (SELECT hash FROM chunks WHERE hash IS NOT NULL)", [])?)
    }

    /// Sessions whose id, cwd or repo contains one of `patterns`.
    pub fn matching(&self, patterns: &[String]) -> anyhow::Result<Vec<String>> {
        if patterns.is_empty() {
            return Ok(vec![]);
        }
        let mut q = self.conn.prepare("SELECT id, coalesce(cwd,''), coalesce(repo,'') FROM sessions")?;
        let rows = q.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?)))?;
        Ok(rows
            .flatten()
            .filter(|(id, cwd, repo)| patterns.iter().any(|p| id.contains(p) || cwd.contains(p) || repo.contains(p)))
            .map(|(id, ..)| id)
            .collect())
    }

    pub fn stats(&self, path: &Path) -> anyhow::Result<Stats> {
        let mut st = Stats::default();
        let mut q = self.conn.prepare(
            "SELECT s.source, count(DISTINCT s.id),
                    (SELECT count(*) FROM events e JOIN sessions x ON x.id = e.session_id WHERE x.source = s.source),
                    (SELECT count(*) FROM chunks c JOIN sessions x ON x.id = c.session_id WHERE x.source = s.source)
             FROM sessions s GROUP BY s.source ORDER BY s.source",
        )?;
        st.sources = q.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?.flatten().collect();
        (st.first, st.last) =
            self.conn.query_row("SELECT coalesce(min(started_at), 0), coalesce(max(updated_at), 0) FROM sessions", [], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })?;
        st.forgotten = self.conn.query_row("SELECT count(*) FROM forgotten", [], |r| r.get(0))?;
        st.dialog_chunks = self.conn.query_row("SELECT count(DISTINCT hash) FROM chunks WHERE kind='dialog'", [], |r| r.get(0))?;
        let mut q = self.conn.prepare(
            "SELECT v.model, count(*), (SELECT count(DISTINCT c.hash) FROM chunks c JOIN vectors w
                                        ON w.hash = c.hash AND w.model = v.model WHERE c.kind = 'dialog')
             FROM vectors v GROUP BY v.model",
        )?;
        st.vectors = q.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?.flatten().collect();
        for suffix in ["", "-wal"] {
            let p = std::path::PathBuf::from(format!("{}{suffix}", path.display()));
            st.bytes += std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
        }
        Ok(st)
    }
}
