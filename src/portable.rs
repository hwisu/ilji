//! One-file copies of the index: `export` for a backup or another machine, `import` to merge one
//! back. The file is plain SQLite; vectors are little-endian and keyed by model and chunk-text
//! hash, so a file made on Linux or macOS (x86-64 or arm64) works unchanged on the other.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, bail};
use rusqlite::{Connection, OpenFlags, OptionalExtension, params};

use crate::collect::SOURCES;
use crate::config::{Config, now};
use crate::ingest;
use crate::model::{Event, Session};
use crate::store::{SCHEMA_VERSION, Store};

#[derive(Debug, Default)]
pub struct Exported {
    pub sessions: i64,
    pub vectors: i64,
    pub bytes: u64,
}

/// Write a consistent snapshot of the index to `out` as a single file (no -wal beside it).
pub fn export(db: &Path, out: &Path, with_vectors: bool, force: bool) -> anyhow::Result<Exported> {
    if out.exists() {
        if same_file(db, out) {
            bail!("{} is the index itself", out.display());
        }
        if !force {
            bail!("{} exists; pass --force to overwrite it", out.display());
        }
        std::fs::remove_file(out)?;
    }
    let src = Store::open_ro(db)?;
    src.conn
        .execute("VACUUM INTO ?1", [out.to_string_lossy()])
        .with_context(|| format!("writing {}", out.display()))?;
    drop(src);
    private(out);
    let dst = Connection::open(out)?;
    if !with_vectors {
        dst.execute("DELETE FROM vectors", [])?;
    }
    // Rollback journal, so the copy stays one file wherever it is opened.
    dst.query_row("PRAGMA journal_mode=DELETE", [], |_| Ok(()))?;
    if !with_vectors {
        dst.execute_batch("VACUUM")?;
    }
    let count = |t: &str| dst.query_row(&format!("SELECT count(*) FROM {t}"), [], |r| r.get::<_, i64>(0));
    let (sessions, vectors) = (count("sessions")?, count("vectors")?);
    drop(dst);
    Ok(Exported { sessions, vectors, bytes: std::fs::metadata(out)?.len() })
}

#[derive(Debug, Default)]
pub struct Imported {
    pub sessions: usize,
    pub chunks: usize,
    pub vectors: usize,
    /// Sessions this index already holds in the same or a newer state.
    pub kept: usize,
    pub old: usize,
    pub excluded: usize,
    pub forgotten: usize,
    /// Forget decisions carried over from the file.
    pub forgets: usize,
}

/// Merge an exported file into the index. Each session goes through this machine's settings
/// (retention, exclusions, tool output, masking, chunk sizes); vectors come along when the
/// chunk text matches, so only chunks cut differently here need embedding again.
pub fn import(db: &Path, file: &Path, cfg: &Config) -> anyhow::Result<Imported> {
    if same_file(db, file) {
        bail!("{} is the index itself", file.display());
    }
    let src = Connection::open_with_flags(file, OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX)
        .with_context(|| format!("opening {}", file.display()))?;
    let schema: Option<String> =
        src.query_row("SELECT value FROM meta WHERE key='schema'", [], |r| r.get(0)).optional().unwrap_or(None);
    if schema.as_deref() != Some(SCHEMA_VERSION) {
        bail!(
            "{} has index schema {}, this ilji uses {SCHEMA_VERSION}; export it again with the same ilji version",
            file.display(),
            schema.as_deref().unwrap_or("(none)")
        );
    }
    let mut store = Store::open(db)?;
    let mut r = Imported::default();

    let forgotten: Vec<String> = src.prepare("SELECT id FROM forgotten")?.query_map([], |x| x.get(0))?.flatten().collect();
    for id in forgotten {
        if !store.is_forgotten(&id) {
            store.forget(&id)?;
            r.forgets += 1;
        }
    }

    let local: HashMap<String, f64> = store
        .conn
        .prepare("SELECT id, coalesce(updated_at, 0) FROM sessions")?
        .query_map([], |x| Ok((x.get(0)?, x.get(1)?)))?
        .flatten()
        .collect();
    let since = if cfg.index.days > 0.0 { now() - cfg.index.days * 86400.0 } else { f64::NEG_INFINITY };
    type Row = (String, String, String, String, String, f64, String, String);
    let sessions: Vec<Row> = src
        .prepare(
            "SELECT id, source, coalesce(title,''), coalesce(cwd,''), coalesce(repo,''), coalesce(updated_at,0),
                    coalesce(signature,''), coalesce(source_ref,'') FROM sessions ORDER BY id",
        )?
        .query_map([], |x| Ok((x.get(0)?, x.get(1)?, x.get(2)?, x.get(3)?, x.get(4)?, x.get(5)?, x.get(6)?, x.get(7)?)))?
        .collect::<Result<_, _>>()?;
    let mut events = src.prepare(
        "SELECT ts, role, coalesce(tool,''), coalesce(text,''), coalesce(output,''), coalesce(source_ref,'')
         FROM events WHERE session_id=?1 ORDER BY seq",
    )?;
    for (id, source, title, cwd, repo, updated, signature, source_ref) in sessions {
        let Some(source) = SOURCES.iter().copied().find(|s| *s == source) else { continue };
        if store.is_forgotten(&id) {
            r.forgotten += 1;
            continue;
        }
        if updated < since {
            r.old += 1;
            continue;
        }
        if local.get(&id).is_some_and(|&here| here >= updated) {
            r.kept += 1;
            continue;
        }
        let events = events
            .query_map([&id], |x| {
                Ok(Event { ts: x.get(0)?, role: x.get(1)?, tool: x.get(2)?, text: x.get(3)?, output: x.get(4)?, source_ref: x.get(5)? })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        let mut s = Session { id, source, title, cwd, repo, source_ref, events };
        if ingest::excluded(&s, &cfg.index.exclude) {
            r.excluded += 1;
            continue;
        }
        if !ingest::prepare(&mut s, cfg) {
            continue;
        }
        r.chunks += store.replace_session(&s, &signature, &cfg.index)?;
        r.sessions += 1;
    }

    let wanted: HashSet<String> = store
        .conn
        .prepare("SELECT DISTINCT hash FROM chunks WHERE hash IS NOT NULL")?
        .query_map([], |x| x.get(0))?
        .flatten()
        .collect();
    let tx = store.conn.transaction()?;
    {
        let mut ins = tx.prepare("INSERT OR IGNORE INTO vectors(model, hash, dim, scale, data) VALUES(?1, ?2, ?3, ?4, ?5)")?;
        let mut q = src.prepare("SELECT model, hash, dim, scale, data FROM vectors")?;
        let mut rows = q.query([])?;
        while let Some(x) = rows.next()? {
            let hash: String = x.get(1)?;
            if wanted.contains(&hash) {
                r.vectors += ins.execute(params![x.get::<_, String>(0)?, hash, x.get::<_, i64>(2)?, x.get::<_, f64>(3)?, x.get::<_, Vec<u8>>(4)?])?;
            }
        }
    }
    tx.commit()?;
    Ok(r)
}

fn same_file(a: &Path, b: &Path) -> bool {
    matches!((a.canonicalize(), b.canonicalize()), (Ok(x), Ok(y)) if x == y)
}

fn private(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed;
    use crate::config::Precision;

    fn session(id: &str, ts: f64) -> Session {
        Session {
            id: id.into(),
            source: "claude",
            title: "배포 실패 원인".into(),
            cwd: "/work/app".into(),
            repo: "app".into(),
            source_ref: "x.jsonl".into(),
            events: vec![
                Event::msg(ts, "user", "배포가 왜 실패했는지 찾아줘, 어제 저녁부터 계속 실패하고 있어", "x.jsonl:1"),
                Event::msg(ts + 1.0, "assistant", "the health check timed out because the port changed to 8081", "x.jsonl:2"),
            ],
        }
    }

    #[test]
    fn export_then_import_moves_sessions_and_vectors() {
        let dir = std::env::temp_dir().join(format!("ilji-portable-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (a, b, file) = (dir.join("a.db"), dir.join("b.db"), dir.join("copy.db"));
        let cfg = Config::default();
        let t = now() - 3600.0;
        {
            let mut sa = Store::open(&a).unwrap();
            sa.replace_session(&session("claude:one", t), "sig", &cfg.index).unwrap();
            sa.forget("claude:gone").unwrap();
            let hash: String = sa.conn.query_row("SELECT hash FROM chunks WHERE kind='dialog'", [], |r| r.get(0)).unwrap();
            let (scale, data) = embed::encode(&[0.6, 0.8], Precision::I8);
            sa.conn
                .execute("INSERT INTO vectors(model, hash, dim, scale, data) VALUES('m', ?1, 2, ?2, ?3)", params![hash, scale, data])
                .unwrap();
        }
        let e = export(&a, &file, true, false).unwrap();
        assert_eq!((e.sessions, e.vectors), (1, 1));
        assert!(export(&a, &file, true, false).is_err(), "refuses to overwrite without --force");
        assert!(!dir.join("copy.db-wal").exists());

        {
            let mut sb = Store::open(&b).unwrap();
            sb.replace_session(&session("claude:gone", t), "sig", &cfg.index).unwrap();
        }
        let r = import(&b, &file, &cfg).unwrap();
        assert_eq!((r.sessions, r.vectors, r.forgets), (1, 1, 1));
        let sb = Store::open_ro(&b).unwrap();
        let ids: Vec<String> = sb.conn.prepare("SELECT id FROM sessions").unwrap().query_map([], |x| x.get(0)).unwrap().flatten().collect();
        assert_eq!(ids, vec!["claude:one".to_string()], "forget decisions travel with the file");
        let hits: i64 = sb.conn.query_row("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH '8081'", [], |x| x.get(0)).unwrap();
        assert_eq!(hits, 1, "imported text is searchable");
        drop(sb);

        let again = import(&b, &file, &cfg).unwrap();
        assert_eq!((again.sessions, again.kept), (0, 1));

        let slim = dir.join("slim.db");
        assert_eq!(export(&a, &slim, false, false).unwrap().vectors, 0);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
