# ilji

*ilji* (일지, "work log") lets coding agents search what you and they did before.

It reads the session history that Codex, Claude Code, OpenCode and pi already keep on your
machine and indexes it as-is: conversation turns and tool runs, no summaries, no extra memory
layer. Agents query it over MCP. You can query it from the shell. Nothing leaves the machine.

- **One index for four agents.** Codex (app database and `.jsonl` / `.jsonl.zst` rollouts),
  Claude Code (`~/.claude/projects`), OpenCode (`opencode*.db`, v1 and v2 tables, read directly)
  and pi (`~/.pi/agent/sessions`).
- **Keyword and meaning, Korean and English.** SQLite FTS5 (BM25) with Korean particle stripping
  and prefix matching, plus multilingual embeddings of the conversation, so a question asked in
  English finds a Korean discussion and the other way round. Each query is routed: lookups with
  identifiers or a few keywords go to keyword search; sentence-like questions use both.
- **Small answers.** Results are grouped by session, with short snippets around the match and a
  character budget, so an agent spends little context per lookup.
- **Secrets masked before storing.** API keys, tokens, JWTs, private keys and passwords in URLs or
  assignments become `[REDACTED]`. Source stores are only ever opened read-only.
- **Every behaviour is a setting.** Sources, tool-output indexing, chunk sizes, snippet length,
  answer budget, Korean handling, ranking, fusion weights, embedding model and precision, and the
  MCP tool set can all be changed or turned off.

Index encryption is next; its setting exists and is inert.

## Requirements

| | |
|---|---|
| OS | Linux (x86-64, arm64) or macOS |
| Build | Rust 1.89 or newer and a C compiler; SQLite is bundled |
| Runtime | One ~8 MB binary. Keyword search needs nothing else |
| Embeddings (optional) | An OpenAI-compatible embeddings endpoint. Default: llama.cpp `llama-server` with harrier-oss-v1-0.6b Q8_0 (640 MB, multilingual), on a GPU or iGPU via Vulkan/Metal, or on the CPU (slower) |
| Disk | About 1.8 MB per session indexed (575 MB for 320 sessions here, vectors included) |
| Memory | ~350 MB peak during a full index; ~10 MB per keyword search, ~35 MB per hybrid search; the embedding server ~200 MB plus the model in GPU memory |

Measured on a Ryzen AI 5 340 laptop (Radeon 840M iGPU): a first index of 90 days (320 sessions,
100k chunks) takes 13–22 s and an incremental pass 0.2 s. Embedding the 32k dialog chunks once
took about an hour on the iGPU (~3,000 characters/s); after that only new chunks are embedded.
A keyword search takes under 10 ms, a hybrid one about 120 ms.

## Install

```sh
cargo install --git https://github.com/hwisu/ilji
ilji index                      # first pass over the last 90 days
```

or from a checkout: `cargo build --release` and use `target/release/ilji`.

For meaning-based search, run one embedding server and fill the vectors:

```sh
ilji setup embed-server         # finds llama-server and the model, or prints how to get them
ilji setup embed-server --apply # installs it as a systemd user service / launchd agent
ilji embed                      # first backfill; later `index` and `watch` embed new chunks
```

Run a single embedding server for all agents: on a shared iGPU a second one can exhaust the
memory the desktop also uses. Without a server, ilji answers with keyword search and says so.

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
ilji search 지난번에 배포가 왜 실패했지        # sessions with snippets
ilji search "why did the deploy fail" --mode semantic
ilji search OOMKilled --kind tool --group chunk -k 5
ilji open 20656                     # a chunk with the turns around it; also e<event id> or a session id
ilji recent --source codex
ilji files src/auth.rs              # sessions that touched a file, newest first
ilji fix "error: linker cc not found"   # what ran after the same error before
ilji stats                          # sizes, per-source counts, embedding coverage
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
ilji config set embed.enabled off          # keyword search only
ilji sources disable opencode
ilji --set search.hybrid=rrf search ...    # one run only
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
| `search.mode` | hybrid | `keyword`, `semantic` or `hybrid` |
| `search.hybrid` | auto | `auto` (route each query), `rrf`, `slots` (tool ranks kept) or `balanced` |
| `search.keyword_weight`, `search.dense_weight`, `search.tool_boost`, `search.rrf_k` | 1, 1, 1, 60 | fusion weights |
| `search.dense_k` | 100 | embedding candidates per query |
| `search.k`, `search.per_session` | 8, 2 | sessions per answer, snippets per session |
| `search.group` | session | `session` or `chunk` |
| `search.snippet`, `search.max_chars` | 500, 12000 | snippet length and answer budget |
| `search.korean_stem`, `search.korean_prefix` | on, on | Korean particle stripping and prefix matching |
| `search.title_weight` | 0.3 | weight of the session title next to the text (0 = text only) |
| `search.fusion` | bm25 | keyword lane: `bm25`, or `rrf` over separate dialog and tool rankings |
| `search.recent_secs` | 300 | hide the last few minutes so a session does not find itself |
| `embed.enabled`, `embed.auto` | on, on | use embeddings; embed new chunks after `index` / `watch` |
| `embed.url`, `embed.model` | `http://127.0.0.1:8767/v1/embeddings`, harrier-oss-v1-0.6b | endpoint and model name |
| `embed.query_prefix`, `embed.passage_prefix` | instruction, "" | text put before queries and passages |
| `embed.precision` | i8 | `i8` (a quarter of the size) or `f32` |
| `embed.batch`, `embed.max_chars`, `embed.min_chars` | 32, 2048, 20 | request size and text limits |
| `embed.skip_imported_tools` | on | leave out tool transcripts Codex stores inside imported messages |
| `serve.tools` | full | `full`: search, open, recent, files, fix; `compact`: one `ilji` tool with a mode |
| `encrypt.mode` | none | reserved: `at-rest`, `he-vectors` |

Vectors are stored per model and per embedding setting, so changing the model or a prefix
re-embeds instead of mixing vectors; `ilji embed --drop <key>` removes an old set.

Paths: the index is in `~/.local/share/ilji` (`ILJI_HOME` or `--home`). Sources follow
`CODEX_HOME`, `CLAUDE_CONFIG_DIR`, `OPENCODE_DATA` / `XDG_DATA_HOME`, `PI_CODING_AGENT_DIR` and
`PI_CODING_AGENT_SESSION_DIR`.

## How well it finds things

On 536 questions generated from one machine's history (four forms per sampled passage; the
target counts as found when its session ranks in the top 10), MRR@10:

| | keyword | semantic | hybrid `rrf` | hybrid `auto` (default) |
|---|---:|---:|---:|---:|
| all | 0.474 | 0.365 | 0.499 | **0.528** |
| conversation passages | 0.505 | 0.515 | 0.610 | **0.622** |
| tool output passages | **0.445** | 0.220 | 0.392 | 0.436 |
| keyword-style queries | **0.756** | 0.389 | 0.654 | **0.756** |
| Korean↔English questions | 0.174 | **0.390** | 0.356 | 0.325 |

The routing rule was written after seeing these forms; on each half of the passages separately
`auto` scored 0.539 and 0.516, against 0.496 and 0.453 for keyword search.

## Privacy

The index holds your conversations and tool output, masked but otherwise verbatim. It stays in
your home directory with owner-only permissions. Masking is pattern-based and can miss things;
`ilji verify-redaction` re-checks the index. The embedding endpoint receives chunk text: keep it
on localhost. Use full-disk encryption (LUKS, FileVault) on any machine that keeps agent history.
Do not commit indexes or session files: this repository holds code only, and its tests build
their inputs at run time.

## License

MIT
