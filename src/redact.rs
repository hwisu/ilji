//! Secret masking applied before anything is stored. Ported from agent-history's redact.py.
//!
//! The `regex` crate has no look-around, so the two assertions of the original are checked in
//! code: a value that is already masked is left alone, and the character before a Secrets
//! Manager ARN must not be part of a word.

use std::sync::LazyLock;

use regex::{Captures, Regex};

pub const MASK: &str = "[REDACTED]";

/// Token formats that are secrets wherever they appear.
static TOKEN_RE: LazyLock<Regex> = LazyLock::new(|| {
    let parts = [
        r"sk-ant-[A-Za-z0-9_\-]{20,}",
        r"sk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_\-]{32,}",
        r"(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{16,}",
        r"gh[pousr]_[A-Za-z0-9]{30,}",
        r"github_pat_[A-Za-z0-9_]{40,}",
        r"glpat-[A-Za-z0-9_\-]{20,}",
        r"xox[abposr]-[A-Za-z0-9\-]{10,}",
        r"AKIA[0-9A-Z]{16}",
        r"ASIA[0-9A-Z]{16}",
        r"AIza[0-9A-Za-z_\-]{35}",
        r"ya29\.[0-9A-Za-z_\-]{20,}",
        r"hf_[A-Za-z0-9]{30,}",
        r"npm_[A-Za-z0-9]{36}",
        r"eyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|\z)",
    ];
    Regex::new(&parts.iter().map(|p| format!("(?:{p})")).collect::<Vec<_>>().join("|")).unwrap()
});

/// key = value / "key": "value": keep the key, mask the value.
static ASSIGN_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?i)((?:password|passwd|secret|api[_\-]?key|access[_\-]?key|private[_\-]?key|auth[_\-]?token|access[_\-]?token|refresh[_\-]?token|client[_\-]?secret|session[_\-]?token)["']?\s*[:=]\s*["']?)([^\s"',;}{]{8,})"#,
    )
    .unwrap()
});

/// What precedes the name in arn:aws:secretsmanager:<region>:<account>:secret:<name>. That name is
/// not the secret and agents search for it.
static ARN_SECRET_HEAD_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"arn:aws(?:-[a-z]+)*:secretsmanager:(?:[a-z0-9\-]+|\*):(?:\d{12}|\*):\z").unwrap()
});
const ARN_SECRET_HEAD_MAX: usize = 80;

static BEARER_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)(\b(?:authorization:\s*)?bearer\s+)([A-Za-z0-9._~+/\-]{16,}=*)").unwrap());
static URL_CRED_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(\b[a-z][a-z0-9+.\-]*://[^/\s:@]+:)([^@\s/]{3,})(@)").unwrap());

fn floor_boundary(s: &str, mut i: usize) -> usize {
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Whether an ASSIGN_RE match is only the name in a real Secrets Manager ARN.
fn is_arn_secret_name(text: &str, caps: &Captures) -> bool {
    let key = caps.get(1).unwrap();
    if key.as_str() != "secret:" {
        return false;
    }
    let start = key.start();
    let lo = floor_boundary(text, start.saturating_sub(ARN_SECRET_HEAD_MAX));
    let window = &text[lo..start];
    let Some(m) = ARN_SECRET_HEAD_RE.find(window) else { return false };
    let before = window[..m.start()].chars().next_back().or_else(|| text[..lo].chars().next_back());
    if before.is_some_and(|c| c.is_alphanumeric() || c == '_' || c == '-') {
        return false;
    }
    !ASSIGN_RE.is_match(caps.get(2).unwrap().as_str())
}

pub fn redact(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    let t = TOKEN_RE.replace_all(text, MASK);
    let t = BEARER_RE.replace_all(&t, |c: &Captures| format!("{}{MASK}", &c[1]));
    let t = URL_CRED_RE.replace_all(&t, |c: &Captures| format!("{}{MASK}{}", &c[1], &c[3]));
    let src = t.as_ref();
    ASSIGN_RE
        .replace_all(src, |c: &Captures| {
            if c[2].starts_with(MASK) || is_arn_secret_name(src, c) {
                c[0].to_string()
            } else {
                format!("{}{MASK}", &c[1])
            }
        })
        .into_owned()
}

/// Values redact() would have masked; empty means the text is clean.
pub fn find_secrets(text: &str) -> Vec<String> {
    let mut hits: Vec<String> = TOKEN_RE.find_iter(text).map(|m| m.as_str().to_string()).collect();
    hits.extend(BEARER_RE.captures_iter(text).filter(|c| &c[2] != MASK).map(|c| c[2].to_string()));
    hits.extend(URL_CRED_RE.captures_iter(text).filter(|c| &c[2] != MASK).map(|c| c[2].to_string()));
    hits.extend(
        ASSIGN_RE
            .captures_iter(text)
            .filter(|c| !c[2].starts_with(MASK) && !is_arn_secret_name(text, c))
            .map(|c| c[2].to_string()),
    );
    hits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks_tokens_and_assignments() {
        for s in [
            "token ghp_abcdefghijklmnopqrstuvwxyz0123456789",
            "AKIAABCDEFGHIJKLMNOP",
            r#""client_secret": "abcd1234efgh5678""#,
            "Authorization: Bearer abcdefghijklmnop1234",
            "postgres://user:hunter2hunter2@db:5432/x",
            "API_KEY=abcd1234efgh5678ijkl",
        ] {
            let r = redact(s);
            assert!(r.contains(MASK), "{s} -> {r}");
            assert!(find_secrets(&r).is_empty(), "{r}");
        }
    }

    #[test]
    fn leaves_prose_alone() {
        let s = "password 정책은 12자 이상, token bucket 알고리즘으로 rate limit";
        assert_eq!(redact(s), s);
    }

    #[test]
    fn keeps_secrets_manager_names() {
        for s in [
            "Resource: arn:aws:secretsmanager:ap-northeast-2:123456789012:secret:prod/db-AbCdEf:password::",
            "arn:aws:secretsmanager:*:*:secret:prod/api-*",
        ] {
            assert_eq!(redact(s), s);
        }
        for (s, value) in [
            ("arn:aws:secretsmanager:ap-northeast-2:123456789012:secret:prod/api:password=hunter2hunter2", "hunter2hunter2"),
            ("x:password=hunter2hunter2", "hunter2hunter2"),
            ("arn:aws:secretsmanager:${REGION}:${ACCOUNT}:secret:prod/api-AbCdEf", "prod/api-AbCdEf"),
        ] {
            assert!(!redact(s).contains(value), "{s} -> {}", redact(s));
        }
    }

    #[test]
    fn already_masked_stays() {
        let s = "password=[REDACTED]";
        assert_eq!(redact(s), s);
        assert!(find_secrets(s).is_empty());
    }
}
