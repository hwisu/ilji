# ilji

*ilji* (일지, "work log") lets coding agents search what you and they did before.

It reads the session history that Codex, Claude Code, OpenCode and pi already keep on your
machine and indexes it as-is: conversation turns and tool runs, no summaries, no extra memory
layer. Agents query it over MCP. You can query it from the shell. Nothing leaves the machine.

- **One index for four agents.** Codex (app database and `.jsonl` / `.jsonl.zst` rollouts),
  Claude Code (`~/.claude/projects`), OpenCode (`opencode*.db`, v1 and v2 tables, read directly)
  and pi (`~/.pi/agent/sessions`).
- **Korean-aware keyword search.** SQLite FTS5 with BM25. Korean query words lose their particles
  and endings and match as prefixes, so 배포가, 배포를 and 배포했던 all find 배포.
- **Small answers.** Results are grouped by session, with short snippets around the match and a
  character budget, so an agent spends little context per lookup.
- **Secrets masked before storing.** API keys, tokens, JWTs, private keys and passwords in URLs or
  assignments become `[REDACTED]`. Source stores are only ever opened read-only.
- **Every behaviour is a setting.** Each source, tool-output indexing, chunk sizes, snippet length,
  answer budget, Korean handling, ranking and the MCP tool set can be changed or turned off.

Status: keyword search. Embeddings (for paraphrased and Korean↔English questions) and index
encryption come next; their settings already exist and are inert.

## Requirements

| | |
|---|---|
| OS | Linux (x86-64, arm64) or macOS |
| Build | Rust 1.89 or newer and a C compiler; SQLite is bundled |
| Runtime | Nothing else: one ~7 MB binary |
| Disk | About 1.5 MB of index per session (480 MB for 320 sessions here; `ilji vacuum` after big rewrites) |
| Memory | ~350 MB peak during a full index, ~12 MB per search |

Measured on a Ryzen AI 5 340 laptop: a first index of 90 days (320 sessions, 100k chunks) takes
13–22 s, an incremental pass 0.2 s, a search 40–70 ms.

## Install

```sh
cargo install --git https://github.com/hwisu/ilji
ilji index                      # first pass over the last 90 days
```

or from a checkout: `cargo build --release` and use `target/release/ilji`.

Connect your agents. `setup` prints what to add; with `--apply` it registers through the
agent's own CLI where there is one (Claude Code, pi):

```sh
ilji setup                      # snippets for claude, codex, opencode and pi
ilji setup claude --apply       # claude mcp add --scope user ilji -- ilji serve
ilji setup pi --apply           # pi mcp add ilji -- ilji serve
ilji setup service --apply      # `ilji watch` as a systemd user service or launchd agent
```

`ilji serve` is read-only; `ilji watch` (or a scheduled `ilji index`) is the only writer.

## Use

```sh
ilji search 배포 실패 원인            # sessions with snippets
ilji search OOMKilled --kind tool --group chunk -k 5
ilji open 20656                     # a chunk with the turns around it; also e<event id> or a session id
ilji recent --source codex
ilji files src/auth.rs              # sessions that touched a file, newest first
ilji fix "error: linker cc not found"   # what ran after the same error before
ilji stats
ilji forget codex:<id>              # drop a session and keep it out
ilji exclude add secret-project     # skip a project by path, repo or id
```

Most commands take `--json`.

## Configure

Settings live in `~/.config/ilji/config.toml`. Later layers win: defaults, the file,
`ILJI_<SECTION>_<KEY>` environment variables, `--set section.key=value`, then command flags.

```sh
ilji config list                    # every setting and its value
ilji config set search.k 5
ilji config set index.tool_output input    # full | input | off
ilji sources disable opencode
ilji --set search.fusion=rrf search ...    # one run only
```

| Setting | Default | What it does |
|---|---|---|
| `sources.{codex,claude,opencode,pi}` | on | which histories to read |
| `index.days` | 90 | retention window |
| `index.tool_output` | full | index tool input and output, input only, or no tool calls |
| `index.dialog_chunk`, `index.tool_window`, `index.tool_overlap` | 1024, 8192, 1024 | chunk resolution in characters |
| `index.max_output` | 262144 | characters of one tool output that are stored |
| `index.redact` | on | mask secrets before storing |
| `index.exclude` | [] | skip sessions whose path, repo or id contains a pattern |
| `search.k`, `search.per_session` | 8, 2 | sessions per answer, snippets per session |
| `search.group` | session | `session` or `chunk` |
| `search.snippet`, `search.max_chars` | 500, 12000 | snippet length and answer budget |
| `search.korean_stem`, `search.korean_prefix` | on, on | Korean particle stripping and prefix matching |
| `search.title_weight` | 0.3 | weight of the session title next to the text (0 = text only) |
| `search.fusion` | bm25 | `bm25`, or `rrf` to rank dialog and tool chunks separately and merge |
| `search.recent_secs` | 300 | hide the last few minutes so a session does not find itself |
| `serve.tools` | full | `full`: search, open, recent, files, fix; `compact`: one `ilji` tool with a mode |
| `encrypt.mode` | none | reserved: `at-rest`, `he-vectors` |

Paths: the index is in `~/.local/share/ilji` (`ILJI_HOME` or `--home`). Sources follow
`CODEX_HOME`, `CLAUDE_CONFIG_DIR`, `OPENCODE_DATA` / `XDG_DATA_HOME`, `PI_CODING_AGENT_DIR` and
`PI_CODING_AGENT_SESSION_DIR`.

## Privacy

The index holds your conversations and tool output, masked but otherwise verbatim. It stays in
your home directory with owner-only permissions. Masking is pattern-based and can miss things;
`ilji verify-redaction` re-checks the index. Use full-disk encryption (LUKS, FileVault) on any
machine that keeps agent history. Do not commit indexes or session files: this repository holds
code only, and its tests build their inputs at run time.

## License

MIT
