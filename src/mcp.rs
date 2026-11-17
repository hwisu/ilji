//! MCP over stdio. Read-only: `ilji watch` (or `ilji index`) is the only writer.
//! `serve.tools = full` exposes one tool per lookup; `compact` exposes a single `ilji` tool with
//! a `mode`, which costs less context every turn.

use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock, Implementation, ServerCapabilities, ServerConfig};
use rmcp::{ErrorData as McpError, ServerHandler, ServiceExt, schemars, tool, tool_handler, tool_router};
use serde::Deserialize;

use crate::config::{Config, Group, Tools};
use crate::render;
use crate::search::{self, OpenArgs, Query};
use crate::store::Store;

const INSTRUCTIONS: &str = "ilji searches this machine's past Codex, Claude Code, OpenCode and pi sessions \
(raw conversation turns and tool runs, secrets masked). Activity from the last few minutes is left out. \
When the GitHub source is on, pull requests (body, reviews, failed checks with log excerpts) and failed \
Actions runs are in it too (source \"github\", ids like github:owner/repo#123); look there before calling gh, \
and use gh only for live status, since the copy is as of the last sync.

When to use
- The user refers to earlier work: \"예전에\", \"지난번\", \"그때\", \"우리가 ~했던\", \"previously\", \"last time\".
- You need a past decision, incident, PR, command, error or setting that is in neither this conversation nor the repository.
- Not for general knowledge or what is already in context.

How to use
1. Search with the user's wording plus concrete cues (names, repo, PR number, error text). Several searches in one message are fine.
2. Hits carry snippets. Open a chunk only when a snippet is cut or you need the turns around it.
3. Stop when each part of the question has evidence; state the source, repo, date and session. If nothing is found, say so.";

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SearchArgs {
    /// Words to look for: the user's wording plus concrete cues.
    pub query: String,
    /// Number of sessions (or chunks with group=chunk) to return.
    pub k: Option<usize>,
    /// codex | claude | opencode | pi | github
    pub source: Option<String>,
    /// Substring of the repo name or working directory.
    pub repo: Option<String>,
    /// dialog | tool
    pub kind: Option<String>,
    /// session (default) or chunk.
    pub group: Option<String>,
    /// Chunk ids already seen.
    pub exclude_ids: Option<Vec<i64>>,
    /// Include the last few minutes.
    pub include_recent: Option<bool>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct OpenArgsIn {
    /// A chunk id, e<event id>, or a session id.
    #[serde(rename = "ref")]
    pub reference: String,
    /// Events before and after.
    pub before: Option<usize>,
    pub after: Option<usize>,
    /// Characters per event, and where to start inside the event (for long outputs).
    pub max_chars: Option<usize>,
    pub offset: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RecentArgs {
    pub n: Option<usize>,
    pub source: Option<String>,
    pub repo: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct FilesArgs {
    /// A file path or name, e.g. src/auth.rs.
    pub path: String,
    pub n: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct FixArgs {
    /// The failing output, verbatim.
    pub error: String,
    pub n: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CompactArgs {
    /// search | open | recent | files | fix
    pub mode: String,
    /// search: words; open: chunk id, e<event id> or session id; files: path; fix: the error text. Empty for recent.
    pub q: Option<String>,
    pub k: Option<usize>,
    pub source: Option<String>,
    pub repo: Option<String>,
    pub exclude_ids: Option<Vec<i64>>,
}

#[derive(Clone)]
struct Core {
    cfg: Config,
}

impl Core {
    fn store(&self) -> Result<Store, McpError> {
        Store::open_ro(&crate::config::db_path()).map_err(|e| McpError::internal_error(format!("{e:#}"), None))
    }

    fn search(&self, a: SearchArgs) -> Result<String, McpError> {
        let store = self.store()?;
        let mut q = Query::new(&a.query, &self.cfg);
        q.k = a.k.unwrap_or(q.k).clamp(1, 30);
        q.source = a.source;
        q.repo = a.repo;
        q.kind = a.kind;
        q.exclude_ids = a.exclude_ids.unwrap_or_default();
        q.include_recent = a.include_recent.unwrap_or(false);
        if let Some(g) = a.group.as_deref() {
            q.group = if g == "chunk" { Group::Chunk } else { Group::Session };
        }
        let r = search::search(&store, &self.cfg, &q).map_err(|e| McpError::invalid_params(format!("{e:#}"), None))?;
        Ok(render::results(&r, q.group, self.cfg.search.max_chars))
    }

    fn open(&self, a: OpenArgsIn) -> Result<String, McpError> {
        let store = self.store()?;
        let args = OpenArgs {
            reference: a.reference,
            before: a.before.unwrap_or(2).min(30),
            after: a.after.unwrap_or(4).min(60),
            max_chars_per_event: a.max_chars.unwrap_or(self.cfg.serve.open_event_chars).clamp(200, 40_000),
            offset: a.offset.unwrap_or(0),
        };
        let text = search::open(&store, &args).map_err(|e| McpError::invalid_params(format!("{e:#}"), None))?;
        Ok(crate::common::head_chars(&text, self.cfg.search.max_chars * 3).to_string())
    }

    fn recent(&self, a: RecentArgs) -> Result<String, McpError> {
        let rows = search::recent(&self.store()?, a.n.unwrap_or(10).min(50), a.source.as_deref(), a.repo.as_deref())
            .map_err(|e| McpError::internal_error(format!("{e:#}"), None))?;
        Ok(render::recent(&rows))
    }

    fn files(&self, a: FilesArgs) -> Result<String, McpError> {
        let r = search::files(&self.store()?, &self.cfg, &a.path, a.n.unwrap_or(10).min(30))
            .map_err(|e| McpError::invalid_params(format!("{e:#}"), None))?;
        Ok(render::results(&r, Group::Session, self.cfg.search.max_chars))
    }

    fn fix(&self, a: FixArgs) -> Result<String, McpError> {
        let rows = search::fix(&self.store()?, &self.cfg, &a.error, a.n.unwrap_or(5).min(20), 4)
            .map_err(|e| McpError::invalid_params(format!("{e:#}"), None))?;
        Ok(render::fix(&a.error, &rows))
    }
}

fn text(r: Result<String, McpError>) -> Result<CallToolResult, McpError> {
    r.map(|t| CallToolResult::success(vec![ContentBlock::text(t)]))
}

fn info() -> ServerConfig {
    ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
        .with_server_info(Implementation::new("ilji", env!("CARGO_PKG_VERSION")))
        .with_instructions(INSTRUCTIONS)
}

#[derive(Clone)]
pub struct FullServer {
    core: Core,
    tool_router: ToolRouter<FullServer>,
}

#[tool_router]
impl FullServer {
    /// Search past sessions by meaning and keywords (Korean and English, either way). Returns sessions with snippets.
    #[tool(annotations(read_only_hint = true))]
    async fn search(&self, Parameters(a): Parameters<SearchArgs>) -> Result<CallToolResult, McpError> {
        let core = self.core.clone();
        text(tokio::task::spawn_blocking(move || core.search(a)).await.map_err(join_err)?)
    }

    /// Read a chunk, an event (e<id>) or a session with the turns around it, full tool output included.
    #[tool(annotations(read_only_hint = true))]
    async fn open(&self, Parameters(a): Parameters<OpenArgsIn>) -> Result<CallToolResult, McpError> {
        let core = self.core.clone();
        text(tokio::task::spawn_blocking(move || core.open(a)).await.map_err(join_err)?)
    }

    /// The most recently updated sessions.
    #[tool(annotations(read_only_hint = true))]
    async fn recent(&self, Parameters(a): Parameters<RecentArgs>) -> Result<CallToolResult, McpError> {
        let core = self.core.clone();
        text(tokio::task::spawn_blocking(move || core.recent(a)).await.map_err(join_err)?)
    }

    /// Sessions that mentioned a file path, newest first.
    #[tool(annotations(read_only_hint = true))]
    async fn files(&self, Parameters(a): Parameters<FilesArgs>) -> Result<CallToolResult, McpError> {
        let core = self.core.clone();
        text(tokio::task::spawn_blocking(move || core.files(a)).await.map_err(join_err)?)
    }

    /// What was run after the same error before.
    #[tool(annotations(read_only_hint = true))]
    async fn fix(&self, Parameters(a): Parameters<FixArgs>) -> Result<CallToolResult, McpError> {
        let core = self.core.clone();
        text(tokio::task::spawn_blocking(move || core.fix(a)).await.map_err(join_err)?)
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for FullServer {
    fn get_info(&self) -> ServerConfig {
        info()
    }
}

#[derive(Clone)]
pub struct CompactServer {
    core: Core,
    tool_router: ToolRouter<CompactServer>,
}

#[tool_router]
impl CompactServer {
    /// Past Codex, Claude Code, OpenCode and pi sessions. mode: search (q = words), open (q = chunk id,
    /// e<event id> or session id), recent, files (q = path), fix (q = error text).
    #[tool(annotations(read_only_hint = true))]
    async fn ilji(&self, Parameters(a): Parameters<CompactArgs>) -> Result<CallToolResult, McpError> {
        let core = self.core.clone();
        text(
            tokio::task::spawn_blocking(move || {
                let q = a.q.unwrap_or_default();
                match a.mode.as_str() {
                    "search" => core.search(SearchArgs {
                        query: q,
                        k: a.k,
                        source: a.source,
                        repo: a.repo,
                        kind: None,
                        group: None,
                        exclude_ids: a.exclude_ids,
                        include_recent: None,
                    }),
                    "open" => core.open(OpenArgsIn { reference: q, before: None, after: None, max_chars: None, offset: None }),
                    "recent" => core.recent(RecentArgs { n: a.k, source: a.source, repo: a.repo }),
                    "files" => core.files(FilesArgs { path: q, n: a.k }),
                    "fix" => core.fix(FixArgs { error: q, n: a.k }),
                    m => Err(McpError::invalid_params(format!("unknown mode {m:?}; use search, open, recent, files or fix"), None)),
                }
            })
            .await
            .map_err(join_err)?,
        )
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for CompactServer {
    fn get_info(&self) -> ServerConfig {
        info()
    }
}

fn join_err(e: tokio::task::JoinError) -> McpError {
    McpError::internal_error(e.to_string(), None)
}

pub async fn serve(cfg: Config) -> anyhow::Result<()> {
    let core = Core { cfg: cfg.clone() };
    match cfg.serve.tools {
        Tools::Full => {
            let server = FullServer { core, tool_router: FullServer::tool_router() }.serve(rmcp::transport::stdio()).await?;
            server.waiting().await?;
        }
        Tools::Compact => {
            let server = CompactServer { core, tool_router: CompactServer::tool_router() }.serve(rmcp::transport::stdio()).await?;
            server.waiting().await?;
        }
    }
    Ok(())
}
