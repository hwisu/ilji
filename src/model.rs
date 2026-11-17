//! What collectors produce: sessions made of events.

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Event {
    pub ts: f64,
    /// "user", "assistant" or "tool".
    pub role: String,
    /// Message text, or the tool input for a tool event.
    pub text: String,
    /// Tool name; empty for messages.
    pub tool: String,
    /// Tool output; empty for messages.
    pub output: String,
    /// Where the event came from in the source store (file:line, db#table/id).
    pub source_ref: String,
}

impl Event {
    pub fn msg(ts: f64, role: &str, text: impl Into<String>, source_ref: impl Into<String>) -> Self {
        Event { ts, role: role.into(), text: text.into(), source_ref: source_ref.into(), ..Default::default() }
    }

    pub fn tool(
        ts: f64,
        name: impl Into<String>,
        input: impl Into<String>,
        output: impl Into<String>,
        source_ref: impl Into<String>,
    ) -> Self {
        Event { ts, role: "tool".into(), tool: name.into(), text: input.into(), output: output.into(), source_ref: source_ref.into() }
    }
}

#[derive(Debug, Clone)]
pub struct Session {
    /// "<source>:<native id>", stable across runs.
    pub id: String,
    pub source: &'static str,
    pub title: String,
    pub cwd: String,
    pub repo: String,
    pub source_ref: String,
    pub events: Vec<Event>,
}

impl Session {
    pub fn started_at(&self) -> f64 {
        self.events.iter().map(|e| e.ts).fold(f64::INFINITY, f64::min)
    }

    pub fn updated_at(&self) -> f64 {
        self.events.iter().map(|e| e.ts).fold(0.0, f64::max)
    }
}

pub type Loader = Box<dyn FnOnce() -> anyhow::Result<Option<Session>> + Send>;

/// A session a collector found, with a cheap change signature. `load` runs only when the
/// signature differs from the stored one.
pub struct Candidate {
    pub id: String,
    pub signature: String,
    pub load: Loader,
}
