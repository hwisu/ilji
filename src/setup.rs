//! `ilji setup`: connect agents, run `watch` as a service, install the embedding server.
//! Each step prints what it would do; `--apply` does it.

use std::path::PathBuf;

use anyhow::{Context, bail};

use crate::config::{self, Config};
use crate::embed;

const AGENTS: [&str; 4] = ["claude", "codex", "opencode", "pi"];

/// `setup` with no target covers the agents; `all` adds the embedding server and the watch
/// service, in the order that lets the first `watch` pass embed what it indexes.
pub fn run(target: Option<&str>, apply: bool, compact: bool) -> anyhow::Result<()> {
    let exe = std::env::current_exe()?.canonicalize()?.display().to_string();
    let mut serve = vec![exe.clone(), "serve".to_string()];
    if compact {
        serve.extend(["--tools".into(), "compact".into()]);
    }
    let targets: Vec<&str> = match target {
        None => AGENTS.to_vec(),
        Some("all") => [&["embed-server", "service"][..], &AGENTS[..]].concat(),
        Some(t) => vec![t],
    };
    let many = targets.len() > 1;
    if std::path::Path::new(&exe).ancestors().any(|a| a.ends_with("target/release") || a.ends_with("target/debug")) {
        println!("# Note: this registers the binary in a cargo build directory; `cargo install --path .` puts it in ~/.cargo/bin.\n");
    }
    let mut failed = Vec::new();
    for t in targets {
        let done = match t {
            "embed-server" => setup_embed_server(apply),
            "service" => setup_service(&exe, apply),
            t if AGENTS.contains(&t) => setup_agent(t, &serve, apply, many),
            other => bail!("unknown target {other:?}; use all, claude, codex, opencode, pi, service or embed-server"),
        };
        match done {
            Err(e) if many => {
                eprintln!("{t}: {e:#}\n");
                failed.push(t);
            }
            other => other?,
        }
    }
    if !failed.is_empty() {
        bail!("setup did not finish for {}", failed.join(", "));
    }
    if !apply {
        println!("# Add --apply to run the above.");
    }
    Ok(())
}

/// Register `ilji serve` with an agent through its own `mcp add`, replacing an older entry.
fn setup_agent(agent: &str, serve: &[String], apply: bool, skip_missing: bool) -> anyhow::Result<()> {
    let words = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<String>>();
    let add = match agent {
        "claude" => words(&["claude", "mcp", "add", "--scope", "user", "ilji", "--"]),
        "codex" => words(&["codex", "mcp", "add", "ilji", "--"]),
        "opencode" => words(&["opencode", "mcp", "add", "--global", "ilji", "--"]),
        _ => words(&["pi", "mcp", "add", "ilji", "--"]),
    };
    let add = [add, serve.to_vec()].concat();
    println!("# {agent}\n{}", shell_join(&add));
    if !apply {
        println!();
        return Ok(());
    }
    if which(agent).is_none() {
        if skip_missing {
            println!("{agent}: not installed, skipped\n");
            return Ok(());
        }
        bail!("{agent} is not on PATH");
    }
    // Claude Code refuses to overwrite an entry; the others replace it.
    if agent == "claude" {
        let _ = std::process::Command::new("claude")
            .args(["mcp", "remove", "--scope", "user", "ilji"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
    }
    let out = std::process::Command::new(&add[0]).args(&add[1..]).output().with_context(|| format!("running {agent}"))?;
    if !out.status.success() {
        bail!("`{agent} mcp add` failed: {}", String::from_utf8_lossy(&out.stderr).trim());
    }
    println!("{agent}: registered\n");
    Ok(())
}

fn shell_join(v: &[String]) -> String {
    v.iter()
        .map(|a| if a.contains([' ', '\'', '"']) { format!("'{}'", a.replace('\'', r"'\''")) } else { a.clone() })
        .collect::<Vec<_>>()
        .join(" ")
}

fn which(name: &str) -> Option<PathBuf> {
    std::env::split_paths(&std::env::var_os("PATH")?).map(|d| d.join(name)).find(|p| p.is_file())
}

/// Where a llama-server and a GGUF model are, if anywhere obvious.
fn find_embed_server(cfg: &Config) -> (Option<PathBuf>, Option<PathBuf>) {
    let on_path = which("llama-server");
    let local = |sub: &str, pred: &dyn Fn(&std::path::Path) -> bool| {
        walkdir::WalkDir::new(config::data_dir().join(sub))
            .max_depth(3)
            .into_iter()
            .flatten()
            .map(|e| e.into_path())
            .find(|p| p.is_file() && pred(p))
    };
    let bin = on_path.or_else(|| local("llama", &|p| p.file_name().is_some_and(|n| n == "llama-server")));
    let model = local("models", &|p| {
        let n = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        n.ends_with(".gguf") && n.starts_with(&cfg.embed.model)
    });
    (bin, model)
}

const LLAMA_URL: &str = "https://github.com/ggml-org/llama.cpp/releases/download/b12065/llama-b12065-bin-ubuntu-vulkan-x64.tar.gz";
const LLAMA_SHA256: &str = "007c2c6f545576f9f8c95d6041a243d6ab8135528a4448e9856591d2cbbac5b0";
const MODEL_FILE: &str = "harrier-oss-v1-0.6b.Q8_0.gguf";
const MODEL_URL: &str = "https://huggingface.co/mradermacher/harrier-oss-v1-0.6b-GGUF/resolve/main/harrier-oss-v1-0.6b.Q8_0.gguf";
const MODEL_SHA256: &str = "ba2c9408f82cfdb73aaf70aaea9f125f3fb24c5e76ce0efb469e1530e5ec53e4";

/// Download with curl and keep the file only if its SHA-256 matches.
fn fetch_verified(url: &str, dest: &std::path::Path, sha256: &str) -> anyhow::Result<()> {
    std::fs::create_dir_all(dest.parent().unwrap())?;
    let part = dest.with_extension("part");
    println!("downloading {url}");
    let ok = std::process::Command::new("curl")
        .args(["-fL", "--progress-bar", "-o"])
        .arg(&part)
        .arg(url)
        .status()
        .context("running curl")?
        .success();
    if !ok {
        let _ = std::fs::remove_file(&part);
        bail!("download failed: {url}");
    }
    let sum = if cfg!(target_os = "macos") {
        std::process::Command::new("shasum").args(["-a", "256"]).arg(&part).output()
    } else {
        std::process::Command::new("sha256sum").arg(&part).output()
    }
    .context("computing SHA-256")?;
    let got = String::from_utf8_lossy(&sum.stdout).split_whitespace().next().unwrap_or("").to_string();
    if got != sha256 {
        let _ = std::fs::remove_file(&part);
        bail!("checksum mismatch for {url}: got {got}, want {sha256}");
    }
    std::fs::rename(&part, dest)?;
    Ok(())
}

/// Fetch the pinned llama-server build and model when they are missing and pinned ones fit.
fn download_embed_server(cfg: &Config, bin: &Option<PathBuf>, model: &Option<PathBuf>) -> anyhow::Result<()> {
    let dir = config::data_dir();
    if bin.is_none() {
        if !(cfg!(target_os = "linux") && cfg!(target_arch = "x86_64")) {
            bail!("no llama-server found; install llama.cpp (macOS: brew install llama.cpp) and run this again");
        }
        let tgz = dir.join("llama/llama-b12065.tar.gz");
        fetch_verified(LLAMA_URL, &tgz, LLAMA_SHA256)?;
        let ok = std::process::Command::new("tar").arg("-xzf").arg(&tgz).arg("-C").arg(dir.join("llama")).status()?.success();
        std::fs::remove_file(&tgz)?;
        if !ok {
            bail!("could not unpack {}", tgz.display());
        }
    }
    if model.is_none() {
        if !MODEL_FILE.starts_with(&cfg.embed.model) {
            bail!("no GGUF file for embed.model {:?} under {}; put one there", cfg.embed.model, dir.join("models").display());
        }
        fetch_verified(MODEL_URL, &dir.join("models").join(MODEL_FILE), MODEL_SHA256)?;
    }
    Ok(())
}

fn setup_embed_server(apply: bool) -> anyhow::Result<()> {
    let cfg = Config::load(&[])?;
    let port = cfg.embed.url.rsplit_once(':').and_then(|(_, r)| r.split('/').next()).unwrap_or("8767").to_string();
    let (mut bin, mut model) = find_embed_server(&cfg);
    if apply && (bin.is_none() || model.is_none()) {
        download_embed_server(&cfg, &bin, &model)?;
        (bin, model) = find_embed_server(&cfg);
    }
    let (Some(bin), Some(model)) = (bin, model) else {
        let dir = config::data_dir();
        println!(
            "# embed-server: needs llama-server (llama.cpp) and a GGUF embedding model; not found in PATH or {dir}.\n\
             # With --apply ilji downloads both (Linux x86-64, Vulkan) and checks their SHA-256. By hand:\n\
             mkdir -p {dir}/llama {dir}/models\n\
             curl -L -o /tmp/llama.tgz {LLAMA_URL}\n\
             echo '{LLAMA_SHA256}  /tmp/llama.tgz' | sha256sum -c - && tar -xzf /tmp/llama.tgz -C {dir}/llama\n\
             curl -L -o {dir}/models/{MODEL_FILE} {MODEL_URL}\n\
             echo '{MODEL_SHA256}  {dir}/models/{MODEL_FILE}' | sha256sum -c -\n\
             # macOS: brew install llama.cpp, then download the model as above.\n",
            dir = dir.display(),
        );
        return Ok(());
    };
    let args = format!(
        "-m {} --embedding --pooling last --host 127.0.0.1 --port {port} -c 8192 -b 8192 -ub 8192 -np 4 -fa on -ngl 999",
        model.display()
    );
    println!("# One embedding server for every agent; a second one on a shared iGPU can exhaust its memory.");
    println!("{} {args}\n", bin.display());
    if cfg!(target_os = "macos") {
        let path = config::home().join("Library/LaunchAgents/io.github.ilji.embed.plist");
        let items: String = std::iter::once(bin.display().to_string())
            .chain(args.split(' ').map(str::to_string))
            .map(|a| format!("<string>{a}</string>"))
            .collect();
        let plist = format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n<plist version=\"1.0\"><dict><key>Label</key><string>io.github.ilji.embed</string><key>ProgramArguments</key><array>{items}</array><key>RunAtLoad</key><true/><key>KeepAlive</key><true/><key>ProcessType</key><string>Background</string></dict></plist>\n"
        );
        println!("# launchd agent ({})\n{plist}", path.display());
        if apply {
            std::fs::create_dir_all(path.parent().unwrap())?;
            std::fs::write(&path, plist)?;
            if embed::reachable(&cfg.embed) {
                println!("wrote {} but did not load it: something already answers at {}\n", path.display(), cfg.embed.url);
            } else {
                std::process::Command::new("launchctl").args(["load", "-w"]).arg(&path).status()?;
                println!("loaded {}\n", path.display());
            }
        }
    } else {
        let path = config::home().join(".config/systemd/user/ilji-embed.service");
        let unit = format!(
            "[Unit]\nDescription=ilji: embedding server (llama.cpp)\n\n[Service]\nExecStart={} {args}\nRestart=on-failure\nRestartSec=30\nNice=5\n\n[Install]\nWantedBy=default.target\n",
            bin.display()
        );
        println!("# systemd user unit ({})\n{unit}", path.display());
        if apply {
            std::fs::create_dir_all(path.parent().unwrap())?;
            std::fs::write(&path, unit)?;
            let systemctl = |args: &[&str]| std::process::Command::new("systemctl").arg("--user").args(args).status();
            systemctl(&["daemon-reload"])?;
            let active =
                std::process::Command::new("systemctl").args(["--user", "--quiet", "is-active", "ilji-embed.service"]).status()?.success();
            if active {
                systemctl(&["restart", "ilji-embed.service"])?;
                println!("restarted ilji-embed.service\n");
            } else if embed::reachable(&cfg.embed) {
                // Another server already answers on this port; two on one iGPU can exhaust its memory.
                systemctl(&["enable", "ilji-embed.service"])?;
                println!(
                    "enabled ilji-embed.service but did not start it: something already answers at {}.\n\
                     Stop that server, then: systemctl --user start ilji-embed.service\n",
                    cfg.embed.url
                );
            } else {
                systemctl(&["enable", "--now", "ilji-embed.service"])?;
                println!("enabled ilji-embed.service\n");
            }
        }
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
            "[Unit]\nDescription=ilji: keep the index of agent sessions fresh\n\n[Service]\nExecStart={exe} watch\nRestart=on-failure\nRestartSec=30\nNice=10\n\n[Install]\nWantedBy=default.target\n"
        );
        println!("# systemd user unit ({})\n{unit}", path.display());
        if apply {
            std::fs::create_dir_all(path.parent().unwrap())?;
            std::fs::write(&path, unit)?;
            std::process::Command::new("systemctl").args(["--user", "daemon-reload"]).status()?;
            std::process::Command::new("systemctl").args(["--user", "enable", "--now", "ilji-watch.service"]).status()?;
            println!("enabled ilji-watch.service (it indexes, then embeds new chunks)\n");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fetch_keeps_only_matching_downloads() {
        let dir = std::env::temp_dir().join(format!("ilji-fetch-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("src.bin");
        std::fs::write(&src, b"hello\n").unwrap();
        let url = format!("file://{}", src.display());
        let good = "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03";
        let bad = dir.join("bad.bin");
        assert!(fetch_verified(&url, &bad, &"0".repeat(64)).is_err());
        assert!(!bad.exists() && !bad.with_extension("part").exists());
        let ok = dir.join("ok.bin");
        fetch_verified(&url, &ok, good).unwrap();
        assert_eq!(std::fs::read(&ok).unwrap(), b"hello\n");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn shell_join_quotes_what_needs_it() {
        let v = ["a".to_string(), "b c".to_string(), "it's".to_string()];
        assert_eq!(shell_join(&v), r"a 'b c' 'it'\''s'");
    }
}
