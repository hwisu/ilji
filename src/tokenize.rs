//! Keyword queries. Text is indexed with SQLite's unicode61 tokenizer, which splits on anything
//! that is not a letter or digit and lowercases. Queries are split the same way; a Korean word
//! loses its trailing particle or ending and becomes a prefix query, so 배포가, 배포를 and
//! 배포했던 all reach 배포. A morphological analyzer can replace this later behind the same API.

/// Longest first. Stripped only when at least two syllables remain.
const ENDINGS: &[&str] = &[
    // verb and adjective endings
    "했었는지",
    "했었던",
    "했었지",
    "했었어",
    "했는지",
    "됐었던",
    "이었던",
    "이었지",
    "이었어",
    "하려고",
    "하려면",
    "하는지",
    "됐는지",
    "였는지",
    "에서부터",
    "으로부터",
    "이라고는",
    "했던",
    "했지",
    "했어",
    "했다",
    "했고",
    "했음",
    "하는",
    "하던",
    "하다",
    "해서",
    "해줘",
    "해봐",
    "하게",
    "하고",
    "하면",
    "한다",
    "됐던",
    "됐다",
    "됐어",
    "되는",
    "되던",
    "되어",
    "돼서",
    "된다",
    "였던",
    "였지",
    "였어",
    "였다",
    "있던",
    "있는",
    "없던",
    "없는",
    "었던",
    "았던",
    "인지",
    "이야",
    // particles
    "에게서",
    "한테서",
    "이라는",
    "이라고",
    "이랑",
    "에서",
    "에게",
    "한테",
    "까지",
    "부터",
    "으로",
    "처럼",
    "보다",
    "마다",
    "조차",
    "라고",
    "라는",
    "이나",
    "이란",
    "에는",
    "에도",
    "로는",
    "던",
    "는지",
    "한",
    "된",
    "와",
    "과",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "의",
    "도",
    "만",
    "로",
    "랑",
    "나",
    "란",
    "요",
    "지",
];

fn is_hangul(c: char) -> bool {
    matches!(c, '\u{AC00}'..='\u{D7A3}' | '\u{3131}'..='\u{318E}')
}

/// Words as unicode61 sees them: runs of letters and digits, lowercased.
pub fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(str::to_lowercase).collect()
}

static BY_LENGTH: std::sync::LazyLock<Vec<&'static str>> = std::sync::LazyLock::new(|| {
    let mut v = ENDINGS.to_vec();
    v.sort_by_key(|e| std::cmp::Reverse(e.chars().count()));
    v
});

pub fn stem(word: &str) -> &str {
    if !word.chars().any(is_hangul) {
        return word;
    }
    for e in BY_LENGTH.iter() {
        if let Some(s) = word.strip_suffix(e)
            && s.chars().count() >= 2
        {
            return s;
        }
    }
    word
}

#[derive(Debug, Clone, Copy)]
pub struct KoreanOpts {
    pub stem: bool,
    pub prefix: bool,
}

impl Default for KoreanOpts {
    fn default() -> Self {
        KoreanOpts { stem: true, prefix: true }
    }
}

/// Terms for an FTS5 MATCH: (term, prefix?).
pub fn query_terms(text: &str, ko: KoreanOpts) -> Vec<(String, bool)> {
    let mut out: Vec<(String, bool)> = Vec::new();
    for w in words(text) {
        let korean = w.chars().any(is_hangul);
        let s = if korean && ko.stem { stem(&w).to_string() } else { w.clone() };
        let prefix = korean && ko.prefix;
        if prefix && s.chars().count() < 2 {
            continue; // one syllable matches nearly everything as a prefix
        }
        let term = (s, prefix);
        if !out.contains(&term) {
            out.push(term);
        }
    }
    out
}

/// An FTS5 query that matches any term. Empty when the text has no searchable word.
pub fn fts_query(text: &str, ko: KoreanOpts) -> String {
    query_terms(text, ko)
        .iter()
        .map(|(t, prefix)| format!("\"{}\"{}", t.replace('"', "\"\""), if *prefix { "*" } else { "" }))
        .collect::<Vec<_>>()
        .join(" OR ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_korean_endings() {
        assert_eq!(stem("배포가"), "배포");
        assert_eq!(stem("배포했던"), "배포");
        assert_eq!(stem("서버를"), "서버");
        assert_eq!(stem("위치"), "위치"); // one syllable would remain
        assert_eq!(stem("아이"), "아이");
        assert_eq!(stem("deploy"), "deploy");
    }

    #[test]
    fn query_matches_unicode61_splitting() {
        assert_eq!(words("PR리뷰를 thread_history v1.2"), vec!["pr리뷰를", "thread", "history", "v1", "2"]);
        let ko = KoreanOpts::default();
        assert_eq!(fts_query("배포가 왜 실패했지? OOMKilled", ko), "\"배포\"* OR \"실패\"* OR \"oomkilled\"");
        assert_eq!(fts_query("??", ko), "");
        let plain = KoreanOpts { stem: false, prefix: false };
        assert_eq!(fts_query("배포가 왜", plain), "\"배포가\" OR \"왜\"");
    }
}
