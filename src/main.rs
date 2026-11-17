//! ilji: search over local coding-agent history (Codex, Claude Code, OpenCode), served over MCP.

mod chunk;
mod collect;
mod common;
mod config;
mod ingest;
mod mcp;
mod model;
mod redact;
mod render;
mod search;
mod store;
mod tokenize;

use std::path::PathBuf;

use anyhow::{Context, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};

use config::{Config, Group, Tools, config_path, db_path};
use store::Store;

#[derive(Parser)]
#[command(name = "ilji", version, about = "Search your past Codex, Claude Code, OpenCode and pi sessions, locally.")]
struct Cli {
    /// Override a setting for this run, e.g. --set search.k=5 (repeatable). See `ilji config list`.
    #[arg(long = "set", global = true, value_name = "KEY=VALUE")]
    set: Vec<String>,
    /// Data directory (default ~/.local/share/ilji; also ILJI_HOME).
    #[arg(long, global = true, value_name = "DIR")]
    home: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Read new or changed sessions into the index.
    Index {
        /// Rewrite every session, changed or not.
        #[arg(long)]
        full: bool,
        /// Only these sources (codex, claude, opencode, pi).
        #[arg(long = "source", value_name = "SOURCE")]
        sources: Vec<String>,
        #[arg(short, long)]
        quiet: bool,
    },
    /// Keep the index fresh: index, then again every INTERVAL seconds. The only writer while it runs.
    Watch {
        #[arg(long, default_value_t = 60)]
        interval: u64,
    },
    /// Keyword search.
    Search(SearchCli),
    /// Show a chunk, an event (e<id>) or a session with the turns around it.
    Open {
        reference: String,
        #[arg(short = 'B', long, default_value_t = 2)]
        before: usize,
        #[arg(short = 'A', long, default_value_t = 4)]
        after: usize,
        /// Characters per event.
        #[arg(long, default_value_t = 4000)]
        chars: usize,
        /// Where to start inside the opened event.
        #[arg(long, default_value_t = 0)]
        offset: usize,
    },
    /// The most recently updated sessions.
    Recent {
        #[arg(short, default_value_t = 15)]
        n: usize,
        #[arg(long)]
        source: Option<String>,
        #[arg(long)]
        repo: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Sessions that mentioned a file path, newest first.
    Files {
        path: String,
        #[arg(short, default_value_t = 10)]
        n: usize,
        #[arg(long)]
        json: bool,
    },
    /// What ran after the same error before.
    Fix {
        error: Vec<String>,
        #[arg(short, default_value_t = 5)]
        n: usize,
        /// Tool calls to show after the error.
        #[arg(long, default_value_t = 4)]
        steps: usize,
        #[arg(long)]
        json: bool,
    },
    /// Index size and per-source counts.
    Stats,
    /// Drop sessions and keep them out of later indexing.
    Forget { sessions: Vec<String> },
    /// Let forgotten sessions be indexed again.
    Unforget { sessions: Vec<String> },
    /// Skip projects whose path, repo or session id contains a pattern.
    Exclude {
        #[command(subcommand)]
        action: ExcludeCmd,
    },
    /// List sources, or turn one on or off.
    Sources {
        #[command(subcommand)]
        action: Option<SourcesCmd>,
    },
    /// Read and change settings in the config file.
    Config {
        #[command(subcommand)]
        action: ConfigCmd,
    },
    /// MCP server on stdio (read-only).
    Serve {
        /// Tool set: full (one tool per lookup) or compact (one tool with a mode).
        #[arg(long, value_enum)]
        tools: Option<ToolsArg>,
    },
    /// Check the index for secrets the masking should have caught.
    VerifyRedaction,
    /// Give space freed by deleted or rewritten sessions back to the disk.
    Vacuum,
    /// Show how to connect an agent (claude, codex, opencode, pi) or run `watch` as a service.
    /// With --apply, registers through the agent's own CLI (claude, pi) or installs the service.
    Setup {
        /// claude | codex | opencode | pi | service (all agents when omitted)
        target: Option<String>,
        #[arg(long)]
        apply: bool,
        #[arg(long, value_enum)]
        tools: Option<ToolsArg>,
    },
}

#[derive(Args)]
struct SearchCli {
    query: Vec<String>,
    #[arg(short)]
    k: Option<usize>,
    #[arg(long)]
    source: Option<String>,
    #[arg(long)]
    repo: Option<String>,
    /// dialog or tool
    #[arg(long)]
    kind: Option<String>,
    #[arg(long, value_enum)]
    group: Option<GroupArg>,
    /// Snippets per session.
    #[arg(long)]
    per_session: Option<usize>,
    /// Characters per snippet.
    #[arg(long)]
    snippet: Option<usize>,
    /// Response budget in characters.
    #[arg(long)]
    max_chars: Option<usize>,
    /// Only sessions active in the last N days.
    #[arg(long)]
    days: Option<f64>,
    #[arg(long)]
    include_recent: bool,
    /// Chunk ids to leave out (comma separated).
    #[arg(long, value_delimiter = ',')]
    exclude_ids: Vec<i64>,
    /// Keep Korean particles and endings in query words.
    #[arg(long)]
    no_stem: bool,
    /// Match Korean words exactly instead of as prefixes.
    #[arg(long)]
    no_prefix: bool,
    #[arg(long)]
    json: bool,
}

#[derive(Clone, Copy, ValueEnum)]
enum GroupArg {
    Session,
    Chunk,
}

#[derive(Clone, Copy, ValueEnum)]
enum ToolsArg {
    Full,
    Compact,
}

#[derive(Subcommand)]
enum ExcludeCmd {
    Add { patterns: Vec<String> },
    Rm { patterns: Vec<String> },
    List,
}

#[derive(Subcommand)]
enum SourcesCmd {
    Enable { name: String },
    Disable { name: String },
}

#[derive(Subcommand)]
enum ConfigCmd {
    /// Every setting with its effective value.
    List,
    Get { key: String },
    Set { key: String, value: String },
    /// Back to the default for one key.
    Unset { key: String },
    /// Where the config file lives.
    Path,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("ilji: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    if let Some(h) = &cli.home {
        // SAFETY: set before any other thread starts.
        unsafe { std::env::set_var("ILJI_HOME", h) };
    }
    let mut cfg = Config::load(&cli.set)?;
    let db = db_path();
    match cli.cmd {
        Cmd::Index { full, sources, quiet } => {
            let mut store = Store::open(&db)?;
            let r = ingest::run(&mut store, &cfg, &sources, full, |m| {
                if !quiet {
                    eprintln!("{m}");
                }
            })?;
            print_report(&r);
        }
        Cmd::Watch { interval } => {
            let mut store = Store::open(&db)?;
            loop {
                let r = ingest::run(&mut store, &cfg, &[], false, |_| {})?;
                let written: usize = r.sources.iter().map(|s| s.written).sum();
                if written > 0 || r.pruned > 0 || !r.errors.is_empty() {
                    print_report(&r);
                }
                std::thread::sleep(std::time::Duration::from_secs(interval.max(5)));
            }
        }
        Cmd::Search(a) => {
            if let Some(v) = a.snippet {
                cfg.search.snippet = v;
            }
            if let Some(v) = a.max_chars {
                cfg.search.max_chars = v;
            }
            if a.no_stem {
                cfg.search.korean_stem = false;
            }
            if a.no_prefix {
                cfg.search.korean_prefix = false;
            }
            cfg.validate()?;
            let store = Store::open_ro(&db)?;
            let mut q = search::Query::new(&a.query.join(" "), &cfg);
            q.k = a.k.unwrap_or(q.k);
            q.source = a.source;
            q.repo = a.repo;
            q.kind = a.kind;
            q.per_session = a.per_session.unwrap_or(q.per_session);
            q.include_recent = a.include_recent;
            q.exclude_ids = a.exclude_ids;
            q.since = a.days.map(|d| config::now() - d * 86400.0);
            if let Some(g) = a.group {
                q.group = match g {
                    GroupArg::Session => Group::Session,
                    GroupArg::Chunk => Group::Chunk,
                };
            }
            let t = std::time::Instant::now();
            let r = search::search(&store, &cfg, &q)?;
            if a.json {
                println!("{}", serde_json::to_string_pretty(&r)?);
            } else {
                print!("{}", render::results(&r, q.group, cfg.search.max_chars));
                eprintln!("({:.0} ms)", t.elapsed().as_secs_f64() * 1000.0);
            }
        }
        Cmd::Open { reference, before, after, chars, offset } => {
            let store = Store::open_ro(&db)?;
            print!("{}", search::open(&store, &search::OpenArgs { reference, before, after, max_chars_per_event: chars, offset })?);
        }
        Cmd::Recent { n, source, repo, json } => {
            let rows = search::recent(&Store::open_ro(&db)?, n, source.as_deref(), repo.as_deref())?;
            if json { println!("{}", serde_json::to_string_pretty(&rows)?) } else { print!("{}", render::recent(&rows)) }
        }
        Cmd::Files { path, n, json } => {
            let r = search::files(&Store::open_ro(&db)?, &cfg, &path, n)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&r)?)
            } else {
                print!("{}", render::results(&r, Group::Session, cfg.search.max_chars))
            }
        }
        Cmd::Fix { error, n, steps, json } => {
            let error = error.join(" ");
            let rows = search::fix(&Store::open_ro(&db)?, &cfg, &error, n, steps)?;
            if json { println!("{}", serde_json::to_string_pretty(&rows)?) } else { print!("{}", render::fix(&error, &rows)) }
        }
        Cmd::Stats => {
            let store = Store::open_ro(&db)?;
            print!("{}", render::stats(&store.stats(&db)?, &db));
        }
        Cmd::Forget { sessions } => {
            let mut store = Store::open(&db)?;
            for s in sessions {
                let existed = store.forget(&s)?;
                println!("{s}: {}", if existed { "removed and forgotten" } else { "not indexed; will stay out" });
            }
        }
        Cmd::Unforget { sessions } => {
            let mut store = Store::open(&db)?;
            for s in sessions {
                println!("{s}: {}", if store.unforget(&s)? { "will be indexed again" } else { "was not forgotten" });
            }
        }
        Cmd::Exclude { action } => match action {
            ExcludeCmd::List => {
                for p in &cfg.index.exclude {
                    println!("{p}");
                }
            }
            ExcludeCmd::Add { patterns } => {
                edit_list("index.exclude", |v| {
                    for p in &patterns {
                        if !v.contains(p) {
                            v.push(p.clone());
                        }
                    }
                })?;
                if db.exists() {
                    let mut store = Store::open(&db)?;
                    let ids = store.matching(&patterns)?;
                    for id in &ids {
                        store.delete_session(id)?;
                    }
                    println!("excluded; removed {} indexed sessions", ids.len());
                }
            }
            ExcludeCmd::Rm { patterns } => {
                edit_list("index.exclude", |v| v.retain(|p| !patterns.contains(p)))?;
                println!("removed; run `ilji index` to bring those sessions back");
            }
        },
        Cmd::Sources { action } => match action {
            None => {
                for name in collect::SOURCES {
                    let path = collect::root(name);
                    let on = if cfg.source_enabled(name) { "on " } else { "off" };
                    let found = if path.exists() { "" } else { "  (not found)" };
                    println!("{name:<9} {on}  {}{found}", path.display());
                }
            }
            Some(SourcesCmd::Enable { name }) => set_key(&format!("sources.{name}"), "true")?,
            Some(SourcesCmd::Disable { name }) => set_key(&format!("sources.{name}"), "false")?,
        },
        Cmd::Config { action } => match action {
            ConfigCmd::Path => println!("{}", config_path().display()),
            ConfigCmd::List => print!("{}", toml::to_string_pretty(&cfg)?),
            ConfigCmd::Get { key } => {
                let root = toml::Value::try_from(&cfg)?;
                let v = config::lookup(&root, &key).with_context(|| format!("unknown setting {key:?}"))?;
                println!("{v}");
            }
            ConfigCmd::Set { key, value } => set_key(&key, &value)?,
            ConfigCmd::Unset { key } => edit_file(|root| {
                let (section, k) = key.split_once('.').context("use section.key")?;
                if let Some(t) = root.get_mut(section).and_then(|s| s.as_table_mut()) {
                    t.remove(k);
                }
                Ok(())
            })?,
        },
        Cmd::Serve { tools } => {
            if let Some(t) = tools {
                cfg.serve.tools = match t {
                    ToolsArg::Full => Tools::Full,
                    ToolsArg::Compact => Tools::Compact,
                };
            }
            if !db.exists() {
                eprintln!("ilji: no index yet at {}; run `ilji index`", db.display());
            }
            tokio::runtime::Builder::new_multi_thread().enable_all().build()?.block_on(mcp::serve(cfg))?;
        }
        Cmd::VerifyRedaction => {
            let store = Store::open_ro(&db)?;
            let mut q = store.conn.prepare("SELECT session_id, coalesce(text,''), coalesce(output,'') FROM events")?;
            let mut rows = q.query([])?;
            let (mut found, mut sessions) = (0usize, std::collections::BTreeSet::new());
            while let Some(r) = rows.next()? {
                let n = redact::find_secrets(&r.get::<_, String>(1)?).len() + redact::find_secrets(&r.get::<_, String>(2)?).len();
                if n > 0 {
                    found += n;
                    sessions.insert(r.get::<_, String>(0)?);
                }
            }
            println!("{found} unmasked values in {} sessions", sessions.len());
            for s in sessions.iter().take(20) {
                println!("  {s}");
            }
            if found > 0 {
                bail!("run `ilji index --full` with index.redact = true");
            }
        }
        Cmd::Setup { target, apply, tools } => setup(target.as_deref(), apply, tools)?,
        Cmd::Vacuum => {
            let store = Store::open(&db)?;
            let before = store.stats(&db)?.bytes;
            store.conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE); VACUUM; PRAGMA wal_checkpoint(TRUNCATE);")?;
            let after = store.stats(&db)?.bytes;
            println!("{:.1} MB → {:.1} MB", before as f64 / 1e6, after as f64 / 1e6);
        }
    }
    Ok(())
}

fn setup(target: Option<&str>, apply: bool, tools: Option<ToolsArg>) -> anyhow::Result<()> {
    let exe = std::env::current_exe()?.canonicalize()?.display().to_string();
    let mut serve = vec!["serve".to_string()];
    if matches!(tools, Some(ToolsArg::Compact)) {
        serve.extend(["--tools".into(), "compact".into()]);
    }
    let quoted = |v: &[String]| v.iter().map(|a| if a.contains(' ') { format!("'{a}'") } else { a.clone() }).collect::<Vec<_>>().join(" ");
    let targets: Vec<&str> = match target {
        Some(t) => vec![t],
        None => vec!["claude", "codex", "opencode", "pi"],
    };
    for t in targets {
        let cli: Option<Vec<String>> = match t {
            "claude" => Some([vec!["claude", "mcp", "add", "--scope", "user", "ilji", "--"].into_iter().map(String::from).collect(), vec![exe.clone()], serve.clone()].concat()),
            "pi" => Some([vec!["pi", "mcp", "add", "ilji", "--"].into_iter().map(String::from).collect(), vec![exe.clone()], serve.clone()].concat()),
            _ => None,
        };
        match (t, cli) {
            (_, Some(cmd)) => {
                println!("# {t}\n{}\n", quoted(&cmd));
                if apply {
                    let status = std::process::Command::new(&cmd[0]).args(&cmd[1..]).status()
                        .with_context(|| format!("running {}", cmd[0]))?;
                    println!("{t}: {}", if status.success() { "registered" } else { "the command failed" });
                }
            }
            ("codex", None) => println!(
                "# codex: add to ~/.codex/config.toml\n[mcp_servers.ilji]\ncommand = {:?}\nargs = {:?}\n",
                exe, serve
            ),
            ("opencode", None) => println!(
                "# opencode: add under \"mcp\" in ~/.config/opencode/opencode.json(c)\n\"ilji\": {{ \"type\": \"local\", \"command\": {}, \"enabled\": true }}\n",
                serde_json::to_string(&[vec![exe.clone()], serve.clone()].concat())?
            ),
            ("service", None) => setup_service(&exe, apply)?,
            (other, None) => bail!("unknown target {other:?}; use claude, codex, opencode, pi or service"),
        }
    }
    if apply && target.is_none() {
        println!("codex and opencode have no registration CLI; add the snippets above by hand.");
    }
    Ok(())
}

fn setup_service(exe: &str, apply: bool) -> anyhow::Result<()> {
    if cfg!(target_os = "macos") {
        let path = config::home().join("Library/LaunchAgents/io.github.ilji.watch.plist");
        let plist = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>io.github.ilji.watch</string>
  <key>ProgramArguments</key><array><string>{exe}</string><string>watch</string></array>
  <key>RunAtLoad</key><true/><key>KeepAlive</key><true/>
  <key>ProcessType</key><string>Background</string>
</dict></plist>
"#
        );
        println!("# launchd agent ({})\n{plist}", path.display());
        if apply {
            std::fs::create_dir_all(path.parent().unwrap())?;
            std::fs::write(&path, plist)?;
            std::process::Command::new("launchctl").args(["load", "-w"]).arg(&path).status()?;
            println!("loaded {}", path.display());
        }
    } else {
        let path = config::home().join(".config/systemd/user/ilji-watch.service");
        let unit = format!(
            "[Unit]\nDescription=ilji: keep the agent-history index fresh\n\n[Service]\nExecStart={exe} watch\nRestart=on-failure\nRestartSec=30\nNice=10\n\n[Install]\nWantedBy=default.target\n"
        );
        println!("# systemd user unit ({})\n{unit}", path.display());
        if apply {
            std::fs::create_dir_all(path.parent().unwrap())?;
            std::fs::write(&path, unit)?;
            std::process::Command::new("systemctl").args(["--user", "daemon-reload"]).status()?;
            std::process::Command::new("systemctl").args(["--user", "enable", "--now", "ilji-watch.service"]).status()?;
            println!("enabled ilji-watch.service");
        }
    }
    Ok(())
}

fn print_report(r: &ingest::Report) {
    for s in &r.sources {
        println!("{:<9} found {:>5}  unchanged {:>5}  written {:>5}  chunks {:>7}", s.source, s.found, s.skipped, s.written, s.chunks);
    }
    if r.excluded > 0 {
        println!("excluded {}", r.excluded);
    }
    if r.pruned > 0 {
        println!("pruned {} sessions older than the retention window", r.pruned);
    }
    for e in r.errors.iter().take(10) {
        eprintln!("error: {e}");
    }
    println!("{:.1} s", r.secs);
}

/// Change the config file, then check the result against the defaults before writing it.
fn edit_file(f: impl FnOnce(&mut toml::Value) -> anyhow::Result<()>) -> anyhow::Result<()> {
    let path = config_path();
    let mut root: toml::Value = if path.exists() {
        toml::from_str(&std::fs::read_to_string(&path)?)?
    } else {
        toml::Value::Table(Default::default())
    };
    f(&mut root)?;
    let mut full = toml::Value::try_from(Config::default())?;
    config::merge(&mut full, root.clone());
    let cfg: Config = full.try_into().context("invalid configuration")?;
    cfg.validate()?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&path, toml::to_string_pretty(&root)?)?;
    Ok(())
}

fn set_key(key: &str, value: &str) -> anyhow::Result<()> {
    edit_file(|root| config::set_path(root, key, value))?;
    println!("{key} = {value}  ({})", config_path().display());
    Ok(())
}

fn edit_list(key: &str, f: impl FnOnce(&mut Vec<String>)) -> anyhow::Result<()> {
    let cfg = Config::load(&[])?;
    let root = toml::Value::try_from(&cfg)?;
    let mut items: Vec<String> = config::lookup(&root, key)
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
        .unwrap_or_default();
    f(&mut items);
    edit_file(|root| {
        let (section, k) = key.split_once('.').unwrap();
        let table = root.as_table_mut().unwrap();
        let sec = table.entry(section).or_insert_with(|| toml::Value::Table(Default::default()));
        sec.as_table_mut().unwrap().insert(k.into(), toml::Value::Array(items.into_iter().map(toml::Value::String).collect()));
        Ok(())
    })
}
