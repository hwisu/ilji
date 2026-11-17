//! Chunking: ~1024-character dialog chunks within a turn, one observation per tool call.
//!
//! Long tool observations are split into 8192-character windows overlapping by 1024. A tool
//! window is stored as a span into its event's text, not copied. Lengths are in characters.

use crate::config::IndexCfg;
use crate::model::Event;

#[derive(Debug, Clone, PartialEq)]
pub enum Body {
    Dialog(String),
    /// Character span into `tool_text(event)`.
    Tool {
        start: usize,
        len: usize,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    /// Indexes into the session's events.
    pub first: usize,
    pub last: usize,
    pub ts: f64,
    pub body: Body,
}

pub fn tool_text(tool: &str, input: &str, output: &str) -> String {
    let head = format!("[{tool}] {input}");
    let head = head.trim_end();
    if output.is_empty() { head.to_string() } else { format!("{head}\n→ {output}") }
}

/// The characters [start, start+len) of `s`.
pub fn char_slice(s: &str, start: usize, len: usize) -> &str {
    let mut it = s.char_indices().map(|(i, _)| i).chain(std::iter::once(s.len()));
    let a = it.nth(start).unwrap_or(s.len());
    let b = if len == 0 { a } else { it.nth(len - 1).unwrap_or(s.len()) };
    &s[a..b]
}

/// Split at a newline or space near the limit so words stay whole where possible.
pub fn split_text(text: &str, size: usize) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut rest: Vec<char> = text.chars().collect();
    while rest.len() > size {
        let window = &rest[size / 2..size];
        let cut = window
            .iter()
            .rposition(|&c| c == '\n')
            .max(window.iter().rposition(|&c| c == ' '))
            .map(|p| p + size / 2)
            .filter(|&p| p > 0)
            .unwrap_or(size);
        pieces.push(rest[..cut].iter().collect());
        let tail: String = rest[cut..].iter().collect();
        rest = tail.trim_start().chars().collect();
    }
    if !rest.is_empty() {
        pieces.push(rest.into_iter().collect());
    }
    pieces
}

/// Window start offsets for a text of `n` characters.
pub fn windows(n: usize, size: usize, overlap: usize) -> Vec<(usize, usize)> {
    if n <= size {
        return vec![(0, n)];
    }
    let step = size - overlap;
    (0..n - overlap).step_by(step).map(|i| (i, size.min(n - i))).collect()
}

pub fn chunk_events(events: &[Event], cfg: &IndexCfg) -> Vec<Chunk> {
    let (dialog_chunk, tool_window, tool_overlap) = (cfg.dialog_chunk, cfg.tool_window, cfg.tool_overlap);
    let mut chunks = Vec::new();
    let mut buf: Vec<String> = Vec::new();
    let (mut first, mut last, mut size) = (usize::MAX, 0usize, 0usize);

    let flush = |chunks: &mut Vec<Chunk>, buf: &mut Vec<String>, first: &mut usize, size: &mut usize, last: usize| {
        if !buf.is_empty() {
            chunks.push(Chunk { first: *first, last, ts: events[*first].ts, body: Body::Dialog(buf.join("\n")) });
        }
        buf.clear();
        *first = usize::MAX;
        *size = 0;
    };

    for (i, e) in events.iter().enumerate() {
        if e.role == "tool" {
            flush(&mut chunks, &mut buf, &mut first, &mut size, last);
            let n = tool_text(&e.tool, &e.text, &e.output).chars().count();
            for (start, len) in windows(n, tool_window, tool_overlap) {
                chunks.push(Chunk { first: i, last: i, ts: e.ts, body: Body::Tool { start, len } });
            }
            continue;
        }
        if e.role == "user" {
            flush(&mut chunks, &mut buf, &mut first, &mut size, last); // a user message starts a new turn
        }
        for piece in split_text(&format!("{}: {}", e.role, e.text.trim()), dialog_chunk) {
            let plen = piece.chars().count();
            if !buf.is_empty() && size + plen + 1 > dialog_chunk {
                flush(&mut chunks, &mut buf, &mut first, &mut size, last);
            }
            if first == usize::MAX {
                first = i;
            }
            buf.push(piece);
            last = i;
            size += plen + 1;
        }
    }
    flush(&mut chunks, &mut buf, &mut first, &mut size, last);
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_slice_handles_hangul() {
        assert_eq!(char_slice("가나다라마", 1, 3), "나다라");
        assert_eq!(char_slice("abc", 2, 10), "c");
        assert_eq!(char_slice("abc", 5, 1), "");
    }

    #[test]
    fn windows_overlap() {
        assert_eq!(windows(10, 20, 4), vec![(0, 10)]);
        assert_eq!(windows(20, 10, 2), vec![(0, 10), (8, 10), (16, 4)]);
    }

    #[test]
    fn turns_and_tools_split() {
        let ev = vec![
            Event::msg(1.0, "user", "배포가 왜 실패했지?", ""),
            Event::msg(2.0, "assistant", "로그를 보겠습니다.", ""),
            Event::tool(3.0, "Bash", "kubectl logs", "OOMKilled", ""),
            Event::msg(4.0, "assistant", "메모리 부족입니다.", ""),
            Event::msg(5.0, "user", "고쳐줘", ""),
        ];
        let c = chunk_events(&ev, &IndexCfg::default());
        assert_eq!(c.len(), 4);
        assert_eq!(c[0].body, Body::Dialog("user: 배포가 왜 실패했지?\nassistant: 로그를 보겠습니다.".into()));
        assert_eq!((c[1].first, c[1].last), (2, 2));
        assert!(matches!(c[1].body, Body::Tool { start: 0, .. }));
        assert_eq!(c[3].body, Body::Dialog("user: 고쳐줘".into()));
    }

    #[test]
    fn long_text_splits_on_spaces() {
        let text = "word ".repeat(600);
        let pieces = split_text(&text, 1024);
        assert!(pieces.len() >= 3);
        assert!(pieces.iter().all(|p| p.chars().count() <= 1024 && !p.starts_with(' ')));
    }
}
