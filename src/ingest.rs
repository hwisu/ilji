//! Collect → prepare (filter, cap, redact) → store. Loading runs on several threads; only the
//! calling thread writes.

use std::sync::mpsc;
use std::time::Instant;

use crate::collect;
use crate::common::{head_chars, is_self_output};
use crate::config::{Config, ToolOutput, now};
use crate::model::{Candidate, Session};
use crate::redact::redact;
use crate::store::Store;

#[derive(Debug, Default)]
pub struct SourceReport {
    pub source: String,
    pub found: usize,
    pub written: usize,
    pub chunks: usize,
    pub skipped: usize,
}

#[derive(Debug, Default)]
pub struct Report {
    pub sources: Vec<SourceReport>,
    pub pruned: usize,
    pub excluded: usize,
    pub errors: Vec<String>,
    pub secs: f64,
}

/// Apply the index settings to a loaded session. Returns false when nothing is left to store.
pub fn prepare(s: &mut Session, cfg: &Config) -> bool {
    let ic = &cfg.index;
    match ic.tool_output {
        ToolOutput::Off => s.events.retain(|e| e.role != "tool"),
        ToolOutput::Input => s.events.iter_mut().filter(|e| e.role == "tool").for_each(|e| e.output.clear()),
        ToolOutput::Full => {}
    }
    s.events.retain(|e| !(e.role == "tool" && is_self_output(&e.output)));
    for e in &mut s.events {
        let n = e.output.chars().count();
        if n > ic.max_output {
            e.output = format!("{}\n…[{} more characters not stored]", head_chars(&e.output, ic.max_output), n - ic.max_output);
        }
        if ic.redact {
            e.text = redact(&e.text);
            e.output = redact(&e.output);
        }
    }
    if ic.redact {
        s.title = redact(&s.title);
    }
    !s.events.is_empty()
}

pub fn excluded(s: &Session, patterns: &[String]) -> bool {
    patterns.iter().any(|p| s.id.contains(p.as_str()) || s.cwd.contains(p.as_str()) || s.repo.contains(p.as_str()))
}

/// Index every enabled source (or only `only`). With `full`, sessions are written even when unchanged.
pub fn run(store: &mut Store, cfg: &Config, only: &[String], full: bool, mut progress: impl FnMut(&str)) -> anyhow::Result<Report> {
    let t0 = Instant::now();
    // index.days = 0 keeps everything.
    let since = if cfg.index.days > 0.0 { now() - cfg.index.days * 86400.0 } else { 0.0 };
    let fingerprint = cfg.index_fingerprint();
    let stored = store.signatures()?;
    let mut report = Report::default();
    let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(1, 8);

    for source in collect::SOURCES {
        if !cfg.source_enabled(source) || (!only.is_empty() && !only.iter().any(|o| o == source)) {
            continue;
        }
        let mut sr = SourceReport { source: source.to_string(), ..Default::default() };
        let candidates = match collect::scan(source, since) {
            Ok(c) => c,
            Err(e) => {
                report.errors.push(format!("{source}: {e:#}"));
                continue;
            }
        };
        sr.found = candidates.len();
        let todo: Vec<(Candidate, String)> = candidates
            .into_iter()
            .filter_map(|c| {
                let sig = format!("{fingerprint}:{}", c.signature);
                let unchanged = !full && stored.get(&c.id) == Some(&sig);
                (!unchanged && !store.is_forgotten(&c.id)).then_some((c, sig))
            })
            .collect();
        sr.skipped = sr.found - todo.len();
        progress(&format!("{source}: {} sessions, {} to read", sr.found, todo.len()));

        let (tx, rx) = mpsc::channel::<(String, String, anyhow::Result<Option<Session>>)>();
        let queue = std::sync::Mutex::new(todo.into_iter());
        std::thread::scope(|scope| -> anyhow::Result<()> {
            for _ in 0..workers {
                let tx = tx.clone();
                let queue = &queue;
                scope.spawn(move || {
                    loop {
                        let next = queue.lock().unwrap().next();
                        let Some((c, sig)) = next else { break };
                        let mut res = (c.load)();
                        if let Ok(Some(s)) = res.as_mut() {
                            if !prepare(s, cfg) {
                                res = Ok(None);
                            }
                        }
                        if tx.send((c.id, sig, res)).is_err() {
                            break;
                        }
                    }
                });
            }
            drop(tx);
            for (id, sig, res) in rx {
                match res {
                    Ok(Some(s)) if excluded(&s, &cfg.index.exclude) => {
                        store.delete_session(&id)?;
                        report.excluded += 1;
                    }
                    Ok(Some(s)) => {
                        sr.chunks += store.replace_session(&s, &sig, &cfg.index)?;
                        sr.written += 1;
                    }
                    Ok(None) => {}
                    Err(e) => report.errors.push(format!("{id}: {e:#}")),
                }
            }
            Ok(())
        })?;
        progress(&format!("{source}: wrote {} sessions, {} chunks", sr.written, sr.chunks));
        report.sources.push(sr);
    }
    for id in store.matching(&cfg.index.exclude)? {
        store.delete_session(&id)?;
        report.excluded += 1;
    }
    report.pruned = store.prune(since)?;
    report.secs = t0.elapsed().as_secs_f64();
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Event;

    fn session() -> Session {
        Session {
            id: "claude:x".into(),
            source: "claude",
            title: "t".into(),
            cwd: "/w/app".into(),
            repo: "app".into(),
            source_ref: String::new(),
            events: vec![
                Event::msg(1.0, "user", "key API_KEY=abcd1234efgh5678ijkl", ""),
                Event::tool(2.0, "Bash", "ls", "x".repeat(50), ""),
                Event::tool(3.0, "Bash", "ilji search", "3 hits for 'q'\n\n### chunk 1 · ...", ""),
            ],
        }
    }

    #[test]
    fn prepare_applies_switches() {
        let mut cfg = Config::default();
        cfg.index.max_output = 10;
        let mut s = session();
        assert!(prepare(&mut s, &cfg));
        assert_eq!(s.events.len(), 2, "self search output dropped");
        assert!(s.events[0].text.contains("[REDACTED]"));
        assert!(s.events[1].output.contains("40 more characters"));

        cfg.index.tool_output = ToolOutput::Off;
        let mut s = session();
        prepare(&mut s, &cfg);
        assert!(s.events.iter().all(|e| e.role != "tool"));

        cfg.index.tool_output = ToolOutput::Input;
        cfg.index.redact = false;
        let mut s = session();
        prepare(&mut s, &cfg);
        assert!(s.events[0].text.contains("abcd1234efgh5678ijkl"));
        assert!(s.events.iter().filter(|e| e.role == "tool").all(|e| e.output.is_empty()));
    }
}
