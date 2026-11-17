//! GitHub pull requests and Actions runs, read through the `gh` CLI (its login and its TLS, so
//! ilji holds no token and needs no HTTP stack of its own).
//!
//! Opt-in (`sources.github`), and the only source that uses the network. Repositories come from
//! `github.repos`: "auto" picks the GitHub origins of local clones that indexed sessions ran in.
//!
//! - A pull request is one session: body, reviews and comments are dialog (embedded); commits,
//!   changed files and failed checks with a log excerpt are tool events (keyword search).
//! - An Actions run outside pull requests is a session of its own (`github.runs`).
//!
//! Listing is newest first and stops at the first page that holds nothing new, once a repository
//! has been listed back to the retention window. `github.max_calls` caps the API calls of one
//! pass; what it leaves out is listed again next pass, so a first backfill spreads over passes.

use std::collections::{BTreeMap, HashSet};
use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use regex::Regex;
use serde_json::Value;

use crate::common::{iso_ts, str_of};
use crate::config::{GithubCfg, Runs, now};
use crate::model::{Candidate, Event, Session};

use super::ScanCtx;

const SLACK_SECS: f64 = 2.0 * 86400.0;

/// The `gh` runner with a shared call budget.
pub struct Gh {
    budget: AtomicI64,
    timeout: Duration,
}

#[derive(Debug)]
struct OutOfBudget;
impl std::fmt::Display for OutOfBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("github.max_calls reached for this pass")
    }
}
impl std::error::Error for OutOfBudget {}

impl Gh {
    pub fn new(max_calls: usize) -> Gh {
        Gh { budget: AtomicI64::new(max_calls as i64), timeout: Duration::from_secs(90) }
    }

    fn raw(&self, args: &[&str]) -> anyhow::Result<Vec<u8>> {
        if self.budget.fetch_sub(1, Ordering::SeqCst) <= 0 {
            return Err(OutOfBudget.into());
        }
        let mut child = Command::new("gh")
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("running gh (install the GitHub CLI and run `gh auth login`)")?;
        let mut out = child.stdout.take().unwrap();
        let mut err = child.stderr.take().unwrap();
        let reader = std::thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = out.read_to_end(&mut buf);
            buf
        });
        let err_reader = std::thread::spawn(move || {
            let mut buf = String::new();
            let _ = err.read_to_string(&mut buf);
            buf
        });
        let t0 = Instant::now();
        let status = loop {
            if let Some(s) = child.try_wait()? {
                break s;
            }
            if t0.elapsed() > self.timeout {
                let _ = child.kill();
                bail!("gh {} timed out", args.first().unwrap_or(&""));
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        let stdout = reader.join().unwrap_or_default();
        let stderr = err_reader.join().unwrap_or_default();
        if !status.success() {
            if stderr.contains("rate limit") {
                self.budget.store(0, Ordering::SeqCst);
            }
            bail!("gh {}: {}", args.get(1).unwrap_or(&""), stderr.trim());
        }
        Ok(stdout)
    }

    fn remaining(&self) -> usize {
        self.budget.load(Ordering::SeqCst).max(0) as usize
    }

    fn rest(&self, path: &str) -> anyhow::Result<Value> {
        Ok(serde_json::from_slice(&self.raw(&["api", path])?)?)
    }

    fn graphql(&self, query: &str, vars: &[(&str, String)]) -> anyhow::Result<Value> {
        let q = format!("query={query}");
        let mut args = vec!["api", "graphql", "-f", &q];
        let vs: Vec<String> = vars.iter().map(|(k, v)| format!("{k}={v}")).collect();
        for (v, (k, _)) in vs.iter().zip(vars) {
            // Numbers go typed (-F), strings raw (-f) so "123abc" stays a string.
            args.push(if *k == "number" { "-F" } else { "-f" });
            args.push(v);
        }
        let v: Value = serde_json::from_slice(&self.raw(&args)?)?;
        if let Some(e) = v.get("errors").filter(|e| !e.is_null()) {
            bail!("GitHub GraphQL: {e}");
        }
        Ok(v)
    }
}

fn out_of_budget(e: &anyhow::Error) -> bool {
    e.downcast_ref::<OutOfBudget>().is_some()
}

/// A repository and the local clone it was found in (for the session's cwd).
#[derive(Debug, Clone)]
struct Repo {
    owner: String,
    name: String,
    clone: String,
}

impl Repo {
    fn slug(&self) -> String {
        format!("{}/{}", self.owner, self.name)
    }
}

/// "owner/name" from a git remote URL, if it points at github.com.
pub fn slug_of(url: &str) -> Option<(String, String)> {
    let rest = url.trim().split_once("github.com").map(|(_, r)| r)?;
    let rest = rest.trim_start_matches([':', '/']).trim_end_matches('/');
    let rest = rest.strip_suffix(".git").unwrap_or(rest);
    let (owner, name) = rest.split_once('/')?;
    (!owner.is_empty() && !name.is_empty() && !name.contains('/')).then(|| (owner.to_string(), name.to_string()))
}

fn discover(cfg: &GithubCfg, since: f64) -> Vec<Repo> {
    let mut repos: BTreeMap<String, Repo> = BTreeMap::new();
    let mut weight: BTreeMap<String, i64> = BTreeMap::new();
    for r in &cfg.repos {
        if let Some((owner, name)) = r.split_once('/') {
            let repo = Repo { owner: owner.into(), name: name.into(), clone: String::new() };
            repos.insert(repo.slug(), repo);
        }
    }
    if cfg.repos.iter().any(|r| r == "auto") {
        // Busiest first, so a first pass with a small budget fills the repositories used most.
        let cwds: Vec<(String, i64)> = crate::common::open_ro(&crate::config::db_path())
            .ok()
            .and_then(|c| {
                let mut q = c
                    .prepare(
                        "SELECT cwd, count(*) n FROM sessions WHERE source != 'github' AND cwd != '' AND updated_at >= ?1
                         GROUP BY cwd ORDER BY n DESC",
                    )
                    .ok()?;
                let rows = q.query_map([since], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))).ok()?.flatten().collect();
                Some(rows)
            })
            .unwrap_or_default();
        let mut seen = HashSet::new();
        for (cwd, n) in cwds {
            if !std::path::Path::new(&cwd).is_dir() {
                continue;
            }
            let Ok(o) = Command::new("git").args(["-C", &cwd, "remote", "get-url", "origin"]).stderr(Stdio::null()).output() else {
                continue;
            };
            let url = String::from_utf8_lossy(&o.stdout).to_string();
            if let Some((owner, name)) = slug_of(&url) {
                *weight.entry(format!("{owner}/{name}")).or_default() += n;
            }
            if !seen.insert(url.clone()) {
                continue;
            }
            if let Some((owner, name)) = slug_of(&url) {
                let top = Command::new("git")
                    .args(["-C", &cwd, "rev-parse", "--show-toplevel"])
                    .stderr(Stdio::null())
                    .output()
                    .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                    .unwrap_or_default();
                let repo = Repo { owner, name, clone: if top.is_empty() { cwd } else { top } };
                repos.entry(repo.slug()).or_insert(repo);
            }
        }
    }
    let explicit: HashSet<&str> = cfg.repos.iter().map(String::as_str).collect();
    let mut out: Vec<Repo> = repos.into_values().collect();
    out.sort_by_key(|r| (!explicit.contains(r.slug().as_str()), -weight.get(&r.slug()).copied().unwrap_or(0)));
    out
}

/// Per-repository listing progress, kept beside the index.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
struct State {
    last_scan: f64,
    settings: String,
    /// Repositories whose pull requests / runs were all stored back to the retention window.
    prs_done: HashSet<String>,
    runs_done: HashSet<String>,
    /// Repositories GitHub could not find, and until when they are left alone.
    #[serde(default)]
    missing: BTreeMap<String, f64>,
}

fn state_path() -> PathBuf {
    crate::config::data_dir().join("github-state.json")
}

fn load_state() -> State {
    std::fs::read(state_path()).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
}

fn save_state(s: &State) {
    if let Ok(b) = serde_json::to_vec_pretty(s) {
        let _ = std::fs::write(state_path(), b);
    }
}

/// Bump when the GitHub sessions are built differently.
const FORMAT: &str = "2";

/// What changes the stored text; part of every signature, so a change re-reads GitHub sessions only.
fn settings_key(c: &GithubCfg) -> String {
    format!(
        "{FORMAT}.{}{}{}{}{}{}{}",
        c.comments as u8, c.files as u8, c.checks as u8, c.logs as u8, c.log_chars, c.log_context, c.logs_per_pr
    )
}

pub fn scan(ctx: &ScanCtx) -> anyhow::Result<Vec<Candidate>> {
    let cfg = ctx.cfg.github.clone();
    let mut state = load_state();
    let key = settings_key(&cfg);
    if state.settings != key {
        state.prs_done.clear();
        state.runs_done.clear();
        state.settings = key.clone();
    }
    if !ctx.explicit && now() - state.last_scan < cfg.interval_secs as f64 {
        return Ok(vec![]);
    }
    let gh = Arc::new(Gh::new(cfg.max_calls));
    let cfg = Arc::new(cfg);
    let mut out = Vec::new();
    state.missing.retain(|_, until| *until > now());
    for repo in discover(&cfg, ctx.since) {
        if state.missing.contains_key(&repo.slug()) {
            continue;
        }
        // Leave room for loading: a listing may queue what about a third of the budget can load.
        let cap = gh.remaining() / 3;
        if cap == 0 {
            break;
        }
        match list_prs(&gh, &cfg, &repo, ctx, &key, state.prs_done.contains(&repo.slug()), cap) {
            Ok((mut c, complete)) => {
                if complete {
                    state.prs_done.insert(repo.slug());
                }
                out.append(&mut c);
            }
            Err(e) if out_of_budget(&e) => break,
            Err(e) => {
                let msg = format!("{e:#}");
                if msg.contains("Could not resolve to a Repository") || msg.contains("Not Found") {
                    state.missing.insert(repo.slug(), now() + 86400.0);
                    continue;
                }
                eprintln!("github {}: {msg}", repo.slug());
            }
        }
        let cap = gh.remaining() / 3;
        if cfg.runs == Runs::Off || cap == 0 {
            continue;
        }
        match list_runs(&gh, &cfg, &repo, ctx, &key, state.runs_done.contains(&repo.slug()), cap) {
            Ok((mut c, complete)) => {
                if complete {
                    state.runs_done.insert(repo.slug());
                }
                out.append(&mut c);
            }
            Err(e) if out_of_budget(&e) => break,
            Err(e) => eprintln!("github {} runs: {e:#}", repo.slug()),
        }
    }
    state.last_scan = now();
    save_state(&state);
    Ok(out)
}

const LIST_PRS: &str = "query($owner:String!,$name:String!,$cursor:String){repository(owner:$owner,name:$name){
pullRequests(first:100,after:$cursor,orderBy:{field:UPDATED_AT,direction:DESC}){pageInfo{hasNextPage endCursor}
nodes{number updatedAt commits(last:1){nodes{commit{statusCheckRollup{state}}}}}}}}";

/// Candidates for pull requests updated in the window. Returns whether the listing reached the
/// window's end with nothing left to load (the repository is then "done" for later passes).
fn list_prs(
    gh: &Arc<Gh>,
    cfg: &Arc<GithubCfg>,
    repo: &Repo,
    ctx: &ScanCtx,
    key: &str,
    done: bool,
    cap: usize,
) -> anyhow::Result<(Vec<Candidate>, bool)> {
    let mut out = Vec::new();
    let mut cursor: Option<String> = None;
    let mut pending = 0;
    loop {
        let mut vars = vec![("owner", repo.owner.clone()), ("name", repo.name.clone())];
        if let Some(c) = &cursor {
            vars.push(("cursor", c.clone()));
        }
        let v = gh.graphql(LIST_PRS, &vars)?;
        let page = &v["data"]["repository"]["pullRequests"];
        let nodes = page["nodes"].as_array().cloned().unwrap_or_default();
        let (mut fresh, mut past_window) = (0, false);
        for n in &nodes {
            let updated = iso_ts(str_of(n, "updatedAt")).unwrap_or(0.0);
            if updated < ctx.since {
                past_window = true;
                break;
            }
            let number = n["number"].as_i64().unwrap_or(0);
            let rollup = n["commits"]["nodes"][0]["commit"]["statusCheckRollup"]["state"].as_str().unwrap_or("");
            let id = format!("github:{}#{number}", repo.slug());
            let sig = format!("{}|{rollup}|{key}", str_of(n, "updatedAt"));
            if !(ctx.known)(&id, &sig) {
                fresh += 1;
                pending += 1;
            } else if updated > now() - SLACK_SECS {
                fresh += 1;
            }
            let (gh, cfg, repo) = (gh.clone(), cfg.clone(), repo.clone());
            out.push(Candidate { id, signature: sig, load: Box::new(move || skip_budget(load_pr(&gh, &cfg, &repo, number))) });
        }
        let more = page["pageInfo"]["hasNextPage"].as_bool().unwrap_or(false);
        if past_window || !more {
            return Ok((out, pending == 0));
        }
        if pending >= cap {
            return Ok((out, false)); // the rest waits for the next pass
        }
        // Once a repository is done, a page with nothing new or recent ends the listing.
        if done && fresh == 0 {
            return Ok((out, true));
        }
        cursor = page["pageInfo"]["endCursor"].as_str().map(str::to_string);
    }
}

fn skip_budget(r: anyhow::Result<Option<Session>>) -> anyhow::Result<Option<Session>> {
    match r {
        Err(e) if out_of_budget(&e) => Ok(None), // listed again next pass
        other => other,
    }
}

fn list_runs(
    gh: &Arc<Gh>,
    cfg: &Arc<GithubCfg>,
    repo: &Repo,
    ctx: &ScanCtx,
    key: &str,
    done: bool,
    cap: usize,
) -> anyhow::Result<(Vec<Candidate>, bool)> {
    let since_day = chrono::DateTime::from_timestamp(ctx.since as i64, 0).map(|d| d.format("%Y-%m-%d").to_string()).unwrap_or_default();
    let status = if cfg.runs == Runs::Failed { "&status=failure" } else { "" };
    let mut out = Vec::new();
    let mut pending = 0;
    for page in 1..=100 {
        let v = gh.rest(&format!(
            "repos/{}/actions/runs?per_page=100&page={page}&created=%3E%3D{since_day}{status}&exclude_pull_requests=true",
            repo.slug()
        ))?;
        let runs = v["workflow_runs"].as_array().cloned().unwrap_or_default();
        let mut fresh = 0;
        for r in &runs {
            let event = str_of(r, "event");
            // Pull request runs are inside their pull request's session.
            if event.starts_with("pull_request") || r["pull_requests"].as_array().is_some_and(|a| !a.is_empty()) {
                continue;
            }
            let run_id = r["id"].as_i64().unwrap_or(0);
            let id = format!("github:{}/run/{run_id}", repo.slug());
            let sig = format!("{}|{}|{}|{}|{key}", str_of(r, "status"), str_of(r, "conclusion"), str_of(r, "updated_at"), r["run_attempt"]);
            let created = iso_ts(str_of(r, "created_at")).unwrap_or(0.0);
            if !(ctx.known)(&id, &sig) {
                fresh += 1;
                pending += 1;
            } else if created > now() - SLACK_SECS {
                fresh += 1;
            }
            let (gh, cfg, repo, run) = (gh.clone(), cfg.clone(), repo.clone(), r.clone());
            out.push(Candidate { id, signature: sig, load: Box::new(move || skip_budget(load_run(&gh, &cfg, &repo, &run))) });
        }
        if runs.len() < 100 {
            return Ok((out, pending == 0));
        }
        if pending >= cap {
            return Ok((out, false));
        }
        if done && fresh == 0 {
            return Ok((out, true));
        }
    }
    Ok((out, false))
}

const PR: &str = "query($owner:String!,$name:String!,$number:Int!){repository(owner:$owner,name:$name){pullRequest(number:$number){
number title url body createdAt updatedAt closedAt mergedAt state isDraft baseRefName headRefName
author{login __typename} mergedBy{login}
comments(first:50){nodes{author{login __typename} body createdAt}}
reviews(first:20){nodes{author{login __typename} state body submittedAt comments(first:20){nodes{path line body createdAt}}}}
commits(last:15){nodes{commit{abbreviatedOid messageHeadline committedDate
 statusCheckRollup{state contexts(first:25){nodes{__typename
  ... on CheckRun{name conclusion status databaseId detailsUrl completedAt}
  ... on StatusContext{context state description targetUrl createdAt}}}}}}}
files(first:100){nodes{path additions deletions}}}}}";

fn who(author: &Value) -> (String, &'static str) {
    let login = str_of(author, "login").to_string();
    let bot = str_of(author, "__typename") == "Bot" || login.ends_with("[bot]");
    (if login.is_empty() { "ghost".into() } else { login }, if bot { "assistant" } else { "user" })
}

fn load_pr(gh: &Gh, cfg: &GithubCfg, repo: &Repo, number: i64) -> anyhow::Result<Option<Session>> {
    let v = gh.graphql(PR, &[("owner", repo.owner.clone()), ("name", repo.name.clone()), ("number", number.to_string())])?;
    let p = &v["data"]["repository"]["pullRequest"];
    if p.is_null() {
        return Ok(None);
    }
    Ok(Some(pr_session(p, repo, cfg, &mut |job| fetch_log(gh, repo, job, cfg))))
}

/// Newer gh refuses to print a response with terminal escapes (CI logs have colours) unless told to.
fn escapes_flag() -> Option<&'static str> {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let ok = *FLAG.get_or_init(|| {
        Command::new("gh")
            .args(["api", "--help"])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).contains("--allow-escape-sequences"))
            .unwrap_or(false)
    });
    ok.then_some("--allow-escape-sequences")
}

fn fetch_log(gh: &Gh, repo: &Repo, job: i64, cfg: &GithubCfg) -> Option<String> {
    let path = format!("repos/{}/actions/jobs/{job}/logs", repo.slug());
    let mut args = vec!["api"];
    args.extend(escapes_flag());
    args.push(&path);
    let raw = gh.raw(&args).ok()?;
    Some(log_excerpt(&String::from_utf8_lossy(&raw), cfg.log_chars, cfg.log_context))
}

/// A pull request as a session. `log` fetches the excerpt of a failed Actions job.
fn pr_session(p: &Value, repo: &Repo, cfg: &GithubCfg, log: &mut dyn FnMut(i64) -> Option<String>) -> Session {
    let number = p["number"].as_i64().unwrap_or(0);
    let url = str_of(p, "url").to_string();
    let created = iso_ts(str_of(p, "createdAt")).unwrap_or(0.0);
    let updated = iso_ts(str_of(p, "updatedAt")).unwrap_or(created);
    let mut events = Vec::new();
    let (author, role) = who(&p["author"]);
    events.push(Event::msg(
        created,
        role,
        format!(
            "PR #{number} by @{author}: {}\n{} ← {}{}\n\n{}",
            str_of(p, "title"),
            str_of(p, "baseRefName"),
            str_of(p, "headRefName"),
            if p["isDraft"].as_bool() == Some(true) { " (draft)" } else { "" },
            str_of(p, "body").trim()
        ),
        url.clone(),
    ));
    if cfg.comments {
        for c in p["comments"]["nodes"].as_array().into_iter().flatten() {
            let (login, role) = who(&c["author"]);
            let body = str_of(c, "body").trim();
            if !body.is_empty() {
                events.push(Event::msg(iso_ts(str_of(c, "createdAt")).unwrap_or(updated), role, format!("@{login}: {body}"), url.clone()));
            }
        }
        for r in p["reviews"]["nodes"].as_array().into_iter().flatten() {
            let (login, role) = who(&r["author"]);
            let mut text = format!("@{login} reviewed: {}", str_of(r, "state"));
            let body = str_of(r, "body").trim();
            if !body.is_empty() {
                text.push_str(&format!("\n{body}"));
            }
            let inline: Vec<String> = r["comments"]["nodes"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|c| !str_of(c, "body").trim().is_empty())
                .map(|c| {
                    format!(
                        "{}:{} {}",
                        str_of(c, "path"),
                        c["line"].as_i64().map(|l| l.to_string()).unwrap_or_default(),
                        str_of(c, "body").trim()
                    )
                })
                .collect();
            if body.is_empty() && inline.is_empty() && str_of(r, "state") == "COMMENTED" {
                continue;
            }
            for c in inline {
                text.push_str(&format!("\n{c}"));
            }
            events.push(Event::msg(iso_ts(str_of(r, "submittedAt")).unwrap_or(updated), role, text, url.clone()));
        }
    }
    let commits: Vec<&Value> = p["commits"]["nodes"].as_array().into_iter().flatten().map(|n| &n["commit"]).collect();
    if !commits.is_empty() {
        let lines: Vec<String> =
            commits.iter().map(|c| format!("{} {}", str_of(c, "abbreviatedOid"), str_of(c, "messageHeadline"))).collect();
        let ts = commits.last().and_then(|c| iso_ts(str_of(c, "committedDate"))).unwrap_or(updated);
        events.push(Event::tool(ts, "commits", lines.join("\n"), "", url.clone()));
    }
    if cfg.files {
        let files: Vec<String> = p["files"]["nodes"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|f| format!("{} (+{} -{})", str_of(f, "path"), f["additions"], f["deletions"]))
            .collect();
        if !files.is_empty() {
            events.push(Event::tool(created, "files", files.join("\n"), "", url.clone()));
        }
    }
    if cfg.checks {
        let mut logs_left = if cfg.logs { cfg.logs_per_pr } else { 0 };
        // Newest commits first, so the log budget goes to the latest failures.
        for c in commits.iter().rev() {
            let oid = str_of(c, "abbreviatedOid");
            let rollup = &c["statusCheckRollup"];
            let ctxs: Vec<&Value> = rollup["contexts"]["nodes"].as_array().into_iter().flatten().collect();
            let mut tally: BTreeMap<String, usize> = BTreeMap::new();
            for x in &ctxs {
                let st = x["conclusion"].as_str().or(x["state"].as_str()).unwrap_or("PENDING").to_string();
                *tally.entry(st).or_default() += 1;
                if !matches!(x["conclusion"].as_str().or(x["state"].as_str()), Some("FAILURE" | "TIMED_OUT" | "STARTUP_FAILURE" | "ERROR"))
                {
                    continue;
                }
                let name = x["name"].as_str().or(x["context"].as_str()).unwrap_or("");
                let link = x["detailsUrl"].as_str().or(x["targetUrl"].as_str()).unwrap_or("");
                let ts = iso_ts(x["completedAt"].as_str().or(x["createdAt"].as_str()).unwrap_or("")).unwrap_or(updated);
                let job = x["databaseId"].as_i64();
                let excerpt = match job {
                    Some(j) if logs_left > 0 => {
                        logs_left -= 1;
                        log(j).unwrap_or_default()
                    }
                    _ => str_of(x, "description").to_string(),
                };
                events.push(Event::tool(
                    ts,
                    "checks",
                    format!("{oid} {name}: {} {link}", x["conclusion"].as_str().or(x["state"].as_str()).unwrap_or("")),
                    excerpt,
                    url.clone(),
                ));
            }
            if !ctxs.is_empty() && std::ptr::eq(*c, *commits.last().unwrap()) {
                let summary: Vec<String> = tally.iter().map(|(k, n)| format!("{n} {}", k.to_lowercase())).collect();
                events.push(Event::tool(
                    updated,
                    "checks",
                    format!("{oid} checks: {} ({})", str_of(rollup, "state"), summary.join(", ")),
                    "",
                    url.clone(),
                ));
            }
        }
    }
    if let Some(t) = iso_ts(str_of(p, "mergedAt")) {
        let by = str_of(&p["mergedBy"], "login");
        events.push(Event::msg(t, "user", format!("PR #{number} merged into {} by @{by}", str_of(p, "baseRefName")), url.clone()));
    } else if let Some(t) = iso_ts(str_of(p, "closedAt")) {
        events.push(Event::msg(t, "user", format!("PR #{number} closed without merging"), url.clone()));
    }
    events.sort_by(|a, b| a.ts.total_cmp(&b.ts));
    Session {
        id: format!("github:{}#{number}", repo.slug()),
        source: "github",
        title: format!("#{number} {}", str_of(p, "title")),
        cwd: repo.clone.clone(),
        repo: repo.name.clone(),
        source_ref: url,
        events,
    }
}

fn load_run(gh: &Gh, cfg: &GithubCfg, repo: &Repo, run: &Value) -> anyhow::Result<Option<Session>> {
    let run_id = run["id"].as_i64().unwrap_or(0);
    let jobs = gh.rest(&format!("repos/{}/actions/runs/{run_id}/jobs?per_page=100", repo.slug()))?;
    Ok(Some(run_session(run, &jobs, repo, cfg, &mut |job| fetch_log(gh, repo, job, cfg))))
}

/// An Actions run outside pull requests as a session.
fn run_session(run: &Value, jobs: &Value, repo: &Repo, cfg: &GithubCfg, log: &mut dyn FnMut(i64) -> Option<String>) -> Session {
    let run_id = run["id"].as_i64().unwrap_or(0);
    let url = str_of(run, "html_url").to_string();
    let created = iso_ts(str_of(run, "created_at")).unwrap_or(0.0);
    let updated = iso_ts(str_of(run, "updated_at")).unwrap_or(created);
    let conclusion = run["conclusion"].as_str().unwrap_or(str_of(run, "status"));
    let workflow = str_of(run, "name");
    let branch = str_of(run, "head_branch");
    let sha = &str_of(run, "head_sha").chars().take(7).collect::<String>();
    let mut events = vec![Event::msg(
        created,
        "assistant",
        format!("Workflow {workflow} {conclusion} on {branch} ({} event): {}", str_of(run, "event"), str_of(run, "display_title")),
        url.clone(),
    )];
    let mut summary = Vec::new();
    let mut logs_left = if cfg.logs { cfg.logs_per_pr } else { 0 };
    for j in jobs["jobs"].as_array().into_iter().flatten() {
        let jc = j["conclusion"].as_str().unwrap_or(str_of(j, "status"));
        let failed_step = j["steps"]
            .as_array()
            .into_iter()
            .flatten()
            .find(|s| str_of(s, "conclusion") == "failure")
            .map(|s| str_of(s, "name").to_string());
        summary.push(format!("{}: {jc}{}", str_of(j, "name"), failed_step.as_ref().map(|s| format!(" (step: {s})")).unwrap_or_default()));
        if jc == "failure" && logs_left > 0 {
            logs_left -= 1;
            let ts = iso_ts(str_of(j, "completed_at")).unwrap_or(updated);
            let excerpt = j["id"].as_i64().and_then(&mut *log).unwrap_or_default();
            events.push(Event::tool(
                ts,
                "actions-log",
                format!("{} failed{}", str_of(j, "name"), failed_step.map(|s| format!(" at step {s}")).unwrap_or_default()),
                excerpt,
                str_of(j, "html_url"),
            ));
        }
    }
    events.push(Event::tool(
        updated,
        "actions",
        format!("{workflow} #{} on {branch} {sha}: {conclusion}\n{url}", run["run_number"]),
        summary.join("\n"),
        url.clone(),
    ));
    events.sort_by(|a, b| a.ts.total_cmp(&b.ts));
    Session {
        id: format!("github:{}/run/{run_id}", repo.slug()),
        source: "github",
        title: format!("{workflow} · {branch} · {conclusion}"),
        cwd: repo.clone.clone(),
        repo: repo.name.clone(),
        source_ref: url,
        events,
    }
}

static TS: std::sync::LazyLock<Regex> = std::sync::LazyLock::new(|| Regex::new(r"^\d{4}-\d\d-\d\dT[\d:.]+Z ?").unwrap());
static ANSI: std::sync::LazyLock<Regex> = std::sync::LazyLock::new(|| Regex::new(r"\x1b\[[0-9;]*[A-Za-z]").unwrap());
static ERR: std::sync::LazyLock<Regex> = std::sync::LazyLock::new(|| {
    Regex::new(r"##\[error\]|(?i:\berror\b[:\[]|\bfailed\b|\bpanicked at\b|exception\b|exit code [1-9]|npm err!)|\bERROR\b|\bFAIL(ED)?\b|error\[E\d+\]")
        .unwrap()
});

/// The useful part of a CI log: lines that look like errors with `context` lines around them,
/// then the last lines, without timestamps and colour codes, at most `max_chars` characters.
pub fn log_excerpt(raw: &str, max_chars: usize, context: usize) -> String {
    let lines: Vec<String> = raw.lines().map(|l| ANSI.replace_all(&TS.replace(l, ""), "").trim_end().to_string()).collect();
    let n = lines.len();
    let mut keep = vec![false; n];
    for (i, l) in lines.iter().enumerate() {
        if ERR.is_match(l) {
            for k in keep.iter_mut().take((i + context + 1).min(n)).skip(i.saturating_sub(context)) {
                *k = true;
            }
        }
    }
    let tail = 50.min(n);
    let mut out = String::new();
    let mut last: Option<usize> = None;
    let mut errors = String::new();
    for i in (0..n).filter(|&i| keep[i] && i < n - tail) {
        if last.is_some_and(|l| l + 1 != i) {
            errors.push_str("…\n");
        }
        errors.push_str(&lines[i]);
        errors.push('\n');
        last = Some(i);
    }
    let tail_text = lines[n - tail..].join("\n");
    let half = max_chars / 2;
    if !errors.is_empty() {
        out.push_str(crate::common::head_chars(&errors, max_chars.saturating_sub(tail_text.chars().count().min(half))));
        out.push_str("…\n");
    }
    let room = max_chars.saturating_sub(out.chars().count());
    let skip = tail_text.chars().count().saturating_sub(room);
    out.extend(tail_text.chars().skip(skip));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn repo() -> Repo {
        Repo { owner: "acme".into(), name: "app".into(), clone: "/work/app".into() }
    }

    #[test]
    fn slugs_from_remotes() {
        assert_eq!(slug_of("git@github.com:acme/app.git"), Some(("acme".into(), "app".into())));
        assert_eq!(slug_of("https://github.com/acme/app\n"), Some(("acme".into(), "app".into())));
        assert_eq!(slug_of("ssh://git@github.com/acme/app.git"), Some(("acme".into(), "app".into())));
        assert_eq!(slug_of("https://gitlab.com/acme/app.git"), None);
    }

    #[test]
    fn log_excerpt_keeps_errors_and_tail() {
        let mut raw = String::new();
        for i in 0..300 {
            raw.push_str(&format!("2026-10-29T23:57:01.1234567Z \x1b[36mstep line {i}\x1b[0m\n"));
            if i == 100 {
                raw.push_str("2026-10-29T23:57:02.0000000Z ##[error]Process completed with exit code 1.\n");
            }
        }
        let x = log_excerpt(&raw, 4000, 2);
        assert!(x.contains("##[error]Process completed with exit code 1."));
        assert!(x.contains("step line 99") && x.contains("step line 101"));
        assert!(!x.contains("step line 50\n"), "unrelated lines are dropped");
        assert!(x.contains("step line 299"), "the tail is kept");
        assert!(!x.contains("2026-10-29T") && !x.contains('\x1b'));
        assert!(log_excerpt(&raw, 500, 2).chars().count() <= 500);
    }

    #[test]
    fn pull_request_becomes_a_session() {
        let p = json!({
            "number": 7, "title": "Fix retry", "url": "https://github.com/acme/app/pull/7", "body": "재시도 시 중복 생성을 막습니다",
            "createdAt": "2026-10-01T00:00:00Z", "updatedAt": "2026-10-02T00:00:00Z", "mergedAt": "2026-10-02T00:00:00Z",
            "closedAt": null, "isDraft": false, "baseRefName": "develop", "headRefName": "fix/retry",
            "author": {"login": "dev", "__typename": "User"}, "mergedBy": {"login": "lead"},
            "comments": {"nodes": [{"author": {"login": "ci[bot]", "__typename": "Bot"}, "body": "coverage 81%", "createdAt": "2026-10-01T01:00:00Z"}]},
            "reviews": {"nodes": [{"author": {"login": "lead", "__typename": "User"}, "state": "CHANGES_REQUESTED", "body": "",
                "submittedAt": "2026-10-01T02:00:00Z", "comments": {"nodes": [{"path": "src/a.kt", "line": 3, "body": "lock here", "createdAt": "x"}]}}]},
            "commits": {"nodes": [{"commit": {"abbreviatedOid": "abc1234", "messageHeadline": "Add idempotency key", "committedDate": "2026-10-01T03:00:00Z",
                "statusCheckRollup": {"state": "FAILURE", "contexts": {"nodes": [
                    {"__typename": "CheckRun", "name": "build", "conclusion": "FAILURE", "databaseId": 99, "detailsUrl": "https://x/99", "completedAt": "2026-10-01T03:10:00Z"},
                    {"__typename": "CheckRun", "name": "lint", "conclusion": "SUCCESS", "databaseId": 98, "detailsUrl": "https://x/98", "completedAt": "2026-10-01T03:05:00Z"}]}}}}]},
            "files": {"nodes": [{"path": "src/a.kt", "additions": 10, "deletions": 2}]}
        });
        let cfg = GithubCfg::default();
        let mut asked = vec![];
        let s = pr_session(&p, &repo(), &cfg, &mut |j| {
            asked.push(j);
            Some("error: duplicate key".into())
        });
        assert_eq!(s.id, "github:acme/app#7");
        assert_eq!(asked, vec![99], "only the failed job's log is fetched");
        let roles: Vec<&str> = s.events.iter().map(|e| e.role.as_str()).collect();
        assert!(roles.contains(&"assistant"), "bot comments are the assistant side");
        let tools: Vec<&str> = s.events.iter().filter(|e| e.role == "tool").map(|e| e.tool.as_str()).collect();
        assert_eq!(tools.iter().filter(|t| **t == "checks").count(), 2, "one failure and one summary");
        assert!(tools.contains(&"files") && tools.contains(&"commits"));
        assert!(s.events.iter().any(|e| e.output.contains("duplicate key")));
        assert!(s.events.iter().any(|e| e.text.contains("src/a.kt:3 lock here")));
        assert!(s.events.last().unwrap().text.contains("merged into develop"));
    }

    #[test]
    fn run_becomes_a_session() {
        let run = json!({"id": 5, "html_url": "https://github.com/acme/app/actions/runs/5", "created_at": "2026-10-01T00:00:00Z",
            "updated_at": "2026-10-01T00:10:00Z", "conclusion": "failure", "status": "completed", "name": "Deploy", "head_branch": "main",
            "head_sha": "abcdef123", "event": "push", "display_title": "Bump api", "run_number": 42});
        let jobs = json!({"jobs": [{"id": 77, "name": "deploy-dev", "conclusion": "failure", "completed_at": "2026-10-01T00:09:00Z",
            "html_url": "https://x/77", "steps": [{"name": "pulumi up", "conclusion": "failure"}]}]});
        let s = run_session(&run, &jobs, &repo(), &GithubCfg::default(), &mut |_| Some("##[error]policy violation".into()));
        assert_eq!(s.id, "github:acme/app/run/5");
        assert!(s.title.contains("Deploy") && s.title.contains("failure"));
        assert!(s.events.iter().any(|e| e.text.contains("deploy-dev failed at step pulumi up") && e.output.contains("policy violation")));
    }
}
