//! Plain-text output shared by the CLI and the MCP server, kept inside a character budget.

use crate::config::Group;
use crate::search::{FixRow, Results, SessionRow, fmt_time};
use crate::store::Stats;

pub fn results(r: &Results, group: Group, max_chars: usize) -> String {
    let hits: usize = r.sessions.iter().map(|s| s.hits.len()).sum();
    let mut out = match group {
        Group::Session => format!("{hits} hits for '{}' in {} sessions ({} matched)\n", r.query, r.sessions.len(), r.matched),
        Group::Chunk => format!("{hits} hits for '{}'\n", r.query),
    };
    if let Some(n) = &r.note {
        out += &format!("note: {n}\n");
    }
    if r.sessions.is_empty() {
        return out + "\nNo hits. Try other words, a cue from the work (file, error, PR), or include_recent.\n";
    }
    for (shown, s) in r.sessions.iter().enumerate() {
        let mut block = String::new();
        match group {
            Group::Session => {
                block += &format!(
                    "\n## {} · {} · {} · {} · {}\n",
                    s.session,
                    s.source,
                    fmt_time(s.started_at, true),
                    one_line(&s.title, 90),
                    s.repo
                );
                for h in &s.hits {
                    block +=
                        &format!("### chunk {} · {} · {} · score {:.2}\n{}\n", h.chunk, h.kind, fmt_time(h.ts, false), h.score, h.snippet);
                }
            }
            Group::Chunk => {
                let h = &s.hits[0];
                block += &format!(
                    "\n### chunk {} · {} · {} · {} · {} · {} · score {:.2}\n{}\n",
                    h.chunk,
                    h.source,
                    fmt_time(h.ts, true),
                    one_line(&h.title, 90),
                    h.repo,
                    h.kind,
                    h.score,
                    h.snippet
                );
            }
        }
        if out.chars().count() + block.chars().count() > max_chars && shown > 0 {
            out += &format!(
                "\n[{} more results cut at {max_chars} characters; narrow the query or raise max_chars]\n",
                r.sessions.len() - shown
            );
            break;
        }
        out += &block;
    }
    out + "\nRead more with open <chunk id>, e<event id> or a session id.\n"
}

pub fn recent(rows: &[SessionRow]) -> String {
    let mut out = String::new();
    for r in rows {
        out += &format!(
            "{} · {} · {} · {} · {} events · {}\n",
            fmt_time(r.updated_at, true),
            r.id,
            r.source,
            r.repo,
            r.events,
            one_line(&r.title, 80)
        );
    }
    if out.is_empty() { "No sessions.\n".into() } else { out }
}

pub fn fix(error: &str, rows: &[FixRow]) -> String {
    if rows.is_empty() {
        return format!("Nothing ran after '{error}' before, as far as the index knows.\n");
    }
    let mut out = format!("After '{}' in {} sessions:\n", one_line(error, 80), rows.len());
    for r in rows {
        out +=
            &format!("\n## {} · {} · {} (error in chunk {})\n", r.session, fmt_time(r.when, true), one_line(&r.title, 80), r.error_chunk);
        for (i, (ev, tool, input, status)) in r.after.iter().enumerate() {
            out += &format!(
                "{}. [e{ev}] {tool}: {}{}\n",
                i + 1,
                one_line(input, 200),
                if status.is_empty() { String::new() } else { format!("  {status}") }
            );
        }
    }
    out
}

pub fn stats(s: &Stats, db: &std::path::Path) -> String {
    let mut out = format!("index {} · {:.1} MB\n", db.display(), s.bytes as f64 / 1e6);
    if s.last > 0.0 {
        out += &format!("range {} → {}\n", fmt_time(s.first, true), fmt_time(s.last, true));
    }
    out += "source      sessions    events    chunks\n";
    for (src, sess, ev, ch) in &s.sources {
        out += &format!("{src:<10} {sess:>9} {ev:>9} {ch:>9}\n");
    }
    if s.forgotten > 0 {
        out += &format!("forgotten sessions: {}\n", s.forgotten);
    }
    for (model, n, covered) in &s.vectors {
        let pct = if s.dialog_chunks > 0 { 100.0 * *covered as f64 / s.dialog_chunks as f64 } else { 0.0 };
        out += &format!("vectors {model}: {n} ({pct:.0}% of {} dialog chunks)\n", s.dialog_chunks);
    }
    out
}

pub fn one_line(s: &str, n: usize) -> String {
    let flat: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= n { flat } else { format!("{}…", crate::common::head_chars(&flat, n)) }
}
