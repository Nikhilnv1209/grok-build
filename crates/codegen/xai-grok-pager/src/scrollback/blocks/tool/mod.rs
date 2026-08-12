//! Tool call blocks - sum type for different tool types.

mod edit;
mod execute;
pub(crate) mod hook;
mod lifecycle;
pub mod list_dir;
pub(crate) mod memory_search;
mod other;
mod read;
pub mod search;
mod search_tool;
mod use_tool;
mod web_fetch;
mod web_search;

pub use edit::{
    DiffLineOutput, DiffRenderConfig, EDIT_HL_MAX_BYTES, EDIT_HL_MAX_LINES, EditHighlightPhase,
    EditLineStyles, EditToolCallBlock, compute_file_scoped_styles, file_text_within_hl_caps,
    render_diff_hunk_highlighted, render_diff_hunks_highlighted, render_diff_hunks_with_styles,
};
pub use execute::ExecuteToolCallBlock;
pub use hook::{HookPhase, HookRunEntry, HookRunStatus, ToolCallHookData};
pub use lifecycle::LifecycleEventBlock;
pub use list_dir::ListDirToolCallBlock;
pub use memory_search::MemorySearchToolCallBlock;
pub use other::OtherToolCallBlock;
pub use read::{ReadMediaKind, ReadToolCallBlock};
pub use search::{
    SearchFileMatch, SearchInputMeta, SearchLineMatch, SearchOutputMode, SearchToolCallBlock,
};
pub use search_tool::{
    DiscoveredTool, SearchToolCallBlock as IntegrationSearchToolCallBlock, discovered_tool_action,
};
pub use use_tool::UseToolCallBlock;
pub use web_fetch::WebFetchToolCallBlock;
pub use web_search::WebSearchToolCallBlock;

use crate::scrollback::block::{BlockContent, join_searchable};
use crate::scrollback::types::{
    AccentStyle, BlockBackground, BlockContext, BlockLine, BlockOutput, DisplayMode, Selectable,
};
use crate::theme::Theme;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use std::fmt;

/// Max error lines shown in the collapsed preview. Short enough to stay
/// glanceable; the rest is available via the fullscreen viewer.
pub(crate) const COLLAPSED_ERROR_PREVIEW_LINES: usize = 2;

/// How to style output lines in a collapsed preview.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PreviewStyle {
    /// Plain primary text (file content, search results, MCP JSON, etc.).
    Plain,
    /// Dimmed/muted text — used for thinking so the collapsed body matches
    /// the faded look of truncated/streaming thought content.
    Muted,
    /// Terminal-native ANSI/SGR highlighting (execute / list_dir stdout).
    Terminal,
}

/// Standard left accent bar for tool-call blocks.
///
/// Always returns a color so every tool type draws a consistent left border:
/// red on failure, animated while running, green on success. Callers that
/// gate on config (`execute.accent_enabled`) should check that first.
///
/// `has_error` mirrors `error.is_some()` on tool blocks (success = no error).
pub(crate) fn tool_status_accent(has_error: bool, is_running: bool) -> Option<AccentStyle> {
    let theme = Theme::current();
    if has_error {
        Some(AccentStyle::static_color(theme.accent_error))
    } else if is_running {
        Some(AccentStyle::animated(theme.accent_running))
    } else {
        Some(AccentStyle::static_color(theme.accent_success))
    }
}

/// Append the collapsed body (error reason + output preview + footer) to
/// an already-built header line list.
///
/// Order of content:
/// 1. Error reason in red (always shown when present — this is why a block
///    is red; never hide it behind collapse).
/// 2. Output preview (up to `max_output_lines`, with proper highlighting).
/// 3. Truncation hint when more output exists.
///
/// A single panel band frames error + output so the preview reads as one
/// contiguous box under the header. Skipped entirely while the tool is still
/// running (callers should check `!ctx.is_running` first) or when there's
/// nothing to show.
pub(crate) fn append_collapsed_body(
    lines: &mut Vec<BlockLine>,
    theme: &Theme,
    error: Option<&str>,
    output: Option<&str>,
    max_output_lines: usize,
    style: PreviewStyle,
) {
    let has_error = error.is_some_and(|e| !e.trim().is_empty());
    let has_output = output.is_some_and(|o| !o.is_empty()) && max_output_lines > 0;
    if !has_error && !has_output {
        return;
    }

    // Top panel pad — frames the body as a contiguous box under the header.
    lines.push(panel_pad(theme));

    if has_error {
        let err = error.unwrap_or("");
        let err_lines: Vec<&str> = err.lines().collect();
        let total = err_lines.len();
        let shown = total.min(COLLAPSED_ERROR_PREVIEW_LINES);
        let err_style = Style::default()
            .fg(theme.accent_error)
            .add_modifier(Modifier::BOLD);
        for line in err_lines.iter().take(shown) {
            lines.push(
                BlockLine::from(Line::from(Span::styled(format!("  ✗ {line}"), err_style)))
                    .with_panel_background(theme.bg_dark)
                    .with_wrap(crate::scrollback::types::WrapMode::Word),
            );
        }
        if total > shown {
            let remaining = total - shown;
            lines.push(
                BlockLine::from(Line::from(Span::styled(
                    format!("  … {remaining} more error lines — double-click to view"),
                    theme.fg(theme.accent_error),
                )))
                .with_panel_background(theme.bg_dark),
            );
        }
    }

    if has_output {
        // Thin separator between error and output when both present.
        if has_error {
            lines.push(panel_pad(theme));
        }
        let (mut preview, total) = render_output_preview(output, theme, max_output_lines, style);
        lines.append(&mut preview);
        if let Some(hint) = preview_hint_line(theme, max_output_lines, total) {
            lines.push(hint);
        }
    }

    // Bottom panel pad closes the box.
    lines.push(panel_pad(theme));
}

/// Empty panel-band row used as top/bottom padding of the preview box.
fn panel_pad(theme: &Theme) -> BlockLine {
    BlockLine::from(Line::from("")).with_panel_background(theme.bg_dark)
}

/// Cheap height estimate for the body produced by [`append_collapsed_body`]
/// (not including the header line). Must stay in lockstep with that helper
/// so off-screen scroll estimates match on-screen exact heights.
///
/// Returns 0 when nothing would be appended (no error and no output preview).
pub(crate) fn estimate_collapsed_body_rows(
    error: Option<&str>,
    output: Option<&str>,
    max_output_lines: usize,
) -> u16 {
    let has_error = error.is_some_and(|e| !e.trim().is_empty());
    let output_total = output
        .filter(|o| !o.is_empty())
        .map(|o| o.lines().count())
        .unwrap_or(0);
    let has_output = output_total > 0 && max_output_lines > 0;
    if !has_error && !has_output {
        return 0;
    }
    let mut rows: u16 = 1; // top pad
    if has_error {
        let total = error.unwrap_or("").lines().count();
        let shown = total.min(COLLAPSED_ERROR_PREVIEW_LINES);
        rows = rows.saturating_add(shown as u16);
        if total > shown {
            rows = rows.saturating_add(1); // error-more hint
        }
    }
    if has_output {
        if has_error {
            rows = rows.saturating_add(1); // separator pad
        }
        let shown = output_total.min(max_output_lines);
        rows = rows.saturating_add(shown as u16);
        if output_total > shown {
            rows = rows.saturating_add(1); // more-lines hint
        }
    }
    rows.saturating_add(1) // bottom pad
}

/// Render up to `max_lines` of `output` as panel-band `BlockLine`s.
///
/// Returns `(preview_lines, total_source_lines)`. Lines are indented two
/// spaces and given a panel background. With [`PreviewStyle::Terminal`],
/// ANSI SGR colors/styles from the command stream are preserved.
pub(crate) fn render_output_preview(
    output: Option<&str>,
    theme: &Theme,
    max_lines: usize,
    style: PreviewStyle,
) -> (Vec<BlockLine>, usize) {
    let Some(output) = output else {
        return (Vec::new(), 0);
    };
    if max_lines == 0 || output.is_empty() {
        return (Vec::new(), 0);
    }

    match style {
        PreviewStyle::Terminal => render_terminal_preview(output, theme, max_lines),
        PreviewStyle::Plain => render_plain_preview(output, theme, max_lines, false),
        PreviewStyle::Muted => render_plain_preview(output, theme, max_lines, true),
    }
}

fn render_plain_preview(
    output: &str,
    theme: &Theme,
    max_lines: usize,
    muted: bool,
) -> (Vec<BlockLine>, usize) {
    let content_lines: Vec<&str> = output.lines().collect();
    let total = content_lines.len();
    let indent = "  ";
    let text_style = if muted {
        theme.muted()
    } else {
        theme.primary()
    };
    let mut lines = Vec::with_capacity(total.min(max_lines));
    for (i, line) in content_lines.iter().enumerate() {
        if i >= max_lines {
            break;
        }
        // Empty source lines still paint a panel band so the box stays solid.
        let text = if line.is_empty() {
            "  ".to_string()
        } else {
            format!("{indent}{line}")
        };
        lines.push(
            BlockLine::from(Line::from(Span::styled(text, text_style)))
                .with_panel_background(theme.bg_dark),
        );
    }
    (lines, total)
}

fn render_terminal_preview(
    output: &str,
    theme: &Theme,
    max_lines: usize,
) -> (Vec<BlockLine>, usize) {
    // Parse the full stream so ANSI state is correct for early lines, then
    // take the first max_lines of the *rendered* transcript.
    let rendered = crate::render::terminal_output::render_terminal_lines(output, theme.primary());
    let total = rendered.len();
    let mut lines = Vec::with_capacity(total.min(max_lines));
    for rl in rendered.into_iter().take(max_lines) {
        // Indent with a non-selectable spacer so ANSI-colored spans keep
        // their own styles (prepending into the first span would tint it).
        let mut spans = vec![Span::styled("  ".to_string(), theme.primary())];
        spans.extend(rl.line.spans);
        lines.push(BlockLine::styled(Line::from(spans)).with_panel_background(theme.bg_dark));
    }
    (lines, total)
}

/// Build the "… N more lines — double-click to view" hint line that follows a
/// collapsed output preview. Returns `None` when there are no hidden lines
/// (`shown >= total`) or when `total == 0`.
pub(crate) fn preview_hint_line(theme: &Theme, shown: usize, total: usize) -> Option<BlockLine> {
    if total == 0 || shown >= total {
        return None;
    }
    let remaining = total - shown;
    let mut line = BlockLine::from(Line::from(Span::styled(
        format!("  … {remaining} more lines — double-click to view"),
        theme.dim(),
    )))
    .with_panel_background(theme.bg_dark);
    // Hint is chrome, not content — keep it out of text selection.
    line.selectable = Selectable::None;
    Some(line)
}

/// Shared selection-range id for tool-call header lines.
///
/// Headers are single logical selection targets (path/query/url/command);
/// using one id across tool kinds keeps multi-line drag/copy grouping simple.
pub(crate) const TOOL_HEADER_RANGE: u16 = 0;

/// 1-based inclusive line range for display.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineRange {
    /// Start line (1-based).
    pub start: usize,
    /// End line (1-based, inclusive).
    pub end: usize,
}

impl LineRange {
    /// Create a new line range.
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// Format as "(start:end)" for display.
    pub fn display(&self) -> String {
        format!("{}:{}", self.start, self.end)
    }
}

impl fmt::Display for LineRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{}", self.start, self.end)
    }
}

/// Semantic class of a verb-groupable (non-destructive) run member, naming
/// what a folded run of consecutive rows touched: "Read 3 files", "Searched
/// 4 patterns". Most kinds classify tool blocks via
/// [`ToolCallBlock::verb_group_kind`]; `Subagent` classifies subagent
/// lifecycle render blocks, which are not tool calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VerbGroupKind {
    /// Plain file reads.
    File,
    /// Skill reads and `Skill` invocations (distinct noun from plain files).
    Skill,
    /// Pattern searches (grep/glob).
    Search,
    /// Directory listings.
    Dir,
    /// Web fetches.
    WebFetch,
    /// Web searches, including X search.
    WebSearch,
    /// Memory searches.
    MemorySearch,
    /// MCP tool discovery (`search_tool`).
    IntegrationSearch,
    /// Subagent lifecycle rows (`RenderBlock::Subagent`).
    Subagent,
    /// Shell commands. Label-only: commands never fold eagerly
    /// ([`ToolCallBlock::verb_group_kind`] excludes them), but a truncation
    /// header describing hidden rows buckets them ("Ran 6 commands").
    Command,
    /// File edits. Label-only, like [`Self::Command`].
    EditFile,
    /// MCP tool dispatches (`use_tool`). Label-only, like [`Self::Command`].
    McpCall,
    /// Unclassified tools. Label-only, like [`Self::Command`].
    OtherTool,
}

impl VerbGroupKind {
    /// Verb-group row verb: present tense while running, past otherwise.
    pub fn verb(self, running: bool) -> &'static str {
        let (past, present) = match self {
            VerbGroupKind::File | VerbGroupKind::Skill => ("Read", "Reading"),
            VerbGroupKind::Search
            | VerbGroupKind::WebSearch
            | VerbGroupKind::MemorySearch
            | VerbGroupKind::IntegrationSearch => ("Searched", "Searching"),
            VerbGroupKind::Dir => ("Listed", "Listing"),
            VerbGroupKind::WebFetch => ("Fetched", "Fetching"),
            VerbGroupKind::Subagent | VerbGroupKind::Command | VerbGroupKind::OtherTool => {
                ("Ran", "Running")
            }
            VerbGroupKind::EditFile => ("Edited", "Editing"),
            VerbGroupKind::McpCall => ("Called", "Calling"),
        };
        if running { present } else { past }
    }

    /// Verb-group row noun, pluralized by `count`.
    pub fn noun(self, count: usize) -> &'static str {
        let (one, many) = match self {
            VerbGroupKind::File | VerbGroupKind::EditFile => ("file", "files"),
            VerbGroupKind::Skill => ("skill", "skills"),
            VerbGroupKind::Search => ("pattern", "patterns"),
            VerbGroupKind::Dir => ("dir", "dirs"),
            VerbGroupKind::WebFetch | VerbGroupKind::WebSearch => ("website", "websites"),
            VerbGroupKind::MemorySearch => ("memory", "memories"),
            VerbGroupKind::IntegrationSearch | VerbGroupKind::McpCall => ("MCP tool", "MCP tools"),
            VerbGroupKind::Subagent => ("subagent", "subagents"),
            VerbGroupKind::Command => ("command", "commands"),
            VerbGroupKind::OtherTool => ("tool", "tools"),
        };
        if count == 1 { one } else { many }
    }
}

/// Tool call block - a sum type for different tool types.
///
/// BlockContent is manually implemented (not via enum_delegate) so we can
/// intercept `output()` to prepend the tool bullet configured in appearance.
#[derive(Debug, Clone)]
pub enum ToolCallBlock {
    /// Execute a shell command.
    Execute(ExecuteToolCallBlock),
    /// Read a file.
    Read(ReadToolCallBlock),
    /// Edit a file (with diff).
    Edit(EditToolCallBlock),
    /// List directory contents.
    ListDir(ListDirToolCallBlock),
    /// Search/grep for pattern.
    Search(SearchToolCallBlock),
    /// Web fetch (URL content retrieval).
    WebFetch(WebFetchToolCallBlock),
    /// Web search (web search with citations).
    WebSearch(WebSearchToolCallBlock),
    /// MCP integration tool discovery (search_tool).
    IntegrationSearch(IntegrationSearchToolCallBlock),
    /// MCP integration tool dispatch (use_tool).
    UseTool(UseToolCallBlock),
    /// Memory search with structured result display.
    MemorySearch(MemorySearchToolCallBlock),
    /// Skill invocation (user skills / slash commands via the Skill tool).
    Skill(OtherToolCallBlock),
    /// Other/unknown tool types.
    Other(OtherToolCallBlock),
    /// Lifecycle event (e.g. `user_prompt_submit`, `session_start`).
    /// Not a real tool call — skipped by `last_tool_call_entry_id()`.
    Lifecycle(LifecycleEventBlock),
}

/// Delegate to inner variant, with tool bullet prepended to output.
macro_rules! delegate_tool {
    ($self:expr, $method:ident ( $($arg:expr),* )) => {
        match $self {
            ToolCallBlock::Execute(b) => b.$method($($arg),*),
            ToolCallBlock::Read(b) => b.$method($($arg),*),
            ToolCallBlock::Edit(b) => b.$method($($arg),*),
            ToolCallBlock::ListDir(b) => b.$method($($arg),*),
            ToolCallBlock::Search(b) => b.$method($($arg),*),
            ToolCallBlock::WebFetch(b) => b.$method($($arg),*),
            ToolCallBlock::WebSearch(b) => b.$method($($arg),*),
            ToolCallBlock::IntegrationSearch(b) => b.$method($($arg),*),
            ToolCallBlock::UseTool(b) => b.$method($($arg),*),
            ToolCallBlock::MemorySearch(b) => b.$method($($arg),*),
            ToolCallBlock::Skill(b) => b.$method($($arg),*),
            ToolCallBlock::Other(b) => b.$method($($arg),*),
            ToolCallBlock::Lifecycle(b) => b.$method($($arg),*),
        }
    };
}

impl BlockContent for ToolCallBlock {
    fn output(&self, ctx: &BlockContext) -> BlockOutput {
        // Bullet prepending is handled by RenderBlock::output() via has_bullet().
        delegate_tool!(self, output(ctx))
    }

    fn accent(&self, ctx: &BlockContext) -> Option<AccentStyle> {
        delegate_tool!(self, accent(ctx))
    }

    fn bullet(&self, ctx: &BlockContext) -> Option<AccentStyle> {
        delegate_tool!(self, bullet(ctx))
    }

    fn accent_background(&self, ctx: &BlockContext) -> bool {
        delegate_tool!(self, accent_background(ctx))
    }

    fn background(&self, ctx: &BlockContext) -> BlockBackground {
        delegate_tool!(self, background(ctx))
    }

    fn has_vpad(&self, ctx: &BlockContext) -> bool {
        delegate_tool!(self, has_vpad(ctx))
    }

    fn has_raw_mode(&self) -> bool {
        delegate_tool!(self, has_raw_mode())
    }

    fn is_foldable(&self) -> bool {
        delegate_tool!(self, is_foldable())
    }

    fn next_fold_mode(&self, current: DisplayMode, is_running: bool) -> DisplayMode {
        delegate_tool!(self, next_fold_mode(current, is_running))
    }

    fn collapse_mode(&self, is_running: bool) -> DisplayMode {
        delegate_tool!(self, collapse_mode(is_running))
    }

    fn default_display_mode(&self) -> DisplayMode {
        delegate_tool!(self, default_display_mode())
    }

    fn finished_display_mode(&self) -> Option<DisplayMode> {
        delegate_tool!(self, finished_display_mode())
    }

    fn is_selectable(&self) -> bool {
        delegate_tool!(self, is_selectable())
    }

    fn has_bullet(&self, ctx: &BlockContext) -> bool {
        ctx.appearance
            .scrollback
            .blocks
            .tool
            .bullet
            .char()
            .is_some()
    }

    fn is_groupable(&self) -> bool {
        true
    }

    fn image_references(&self) -> &[crate::prompt_images::ScrollbackImageRef] {
        delegate_tool!(self, image_references())
    }

    fn video_references(&self) -> &[crate::prompt_images::ScrollbackVideoRef] {
        delegate_tool!(self, video_references())
    }

    fn inline_media(&self) -> Option<crate::prompt_images::InlineMediaInfo> {
        delegate_tool!(self, inline_media())
    }

    fn inline_open_button(&self) -> Option<(std::path::PathBuf, bool)> {
        delegate_tool!(self, inline_open_button())
    }

    fn preamble(&self, ctx: &BlockContext) -> Option<ratatui::text::Text<'static>> {
        delegate_tool!(self, preamble(ctx))
    }
}

impl ToolCallBlock {
    /// Transfer timing data from another block of the same variant.
    ///
    /// Used when a running block is replaced with its completed version
    /// (e.g., in `handle_tool_call_update` completion path). The new block
    /// inherits `started_at` from the old block so `finish()` can compute
    /// real elapsed time.
    pub fn transfer_timing_from(&mut self, old: &ToolCallBlock) {
        match (self, old) {
            (ToolCallBlock::Execute(new), ToolCallBlock::Execute(old)) => {
                new.started_at = old.started_at;
            }
            (ToolCallBlock::Read(new), ToolCallBlock::Read(old)) => {
                new.started_at = old.started_at;
            }
            (ToolCallBlock::Edit(new), ToolCallBlock::Edit(old)) => {
                new.started_at = old.started_at;
            }
            (ToolCallBlock::Search(new), ToolCallBlock::Search(old)) => {
                new.started_at = old.started_at;
            }
            (ToolCallBlock::ListDir(new), ToolCallBlock::ListDir(old)) => {
                new.started_at = old.started_at;
            }
            (ToolCallBlock::WebFetch(new), ToolCallBlock::WebFetch(old)) => {
                new.started_at = old.started_at;
            }
            (ToolCallBlock::WebSearch(new), ToolCallBlock::WebSearch(old)) => {
                new.started_at = old.started_at;
            }
            (ToolCallBlock::IntegrationSearch(new), ToolCallBlock::IntegrationSearch(old)) => {
                new.started_at = old.started_at;
            }
            (ToolCallBlock::UseTool(new), ToolCallBlock::UseTool(old)) => {
                new.started_at = old.started_at;
            }
            (ToolCallBlock::Skill(new), ToolCallBlock::Skill(old)) => {
                new.started_at = old.started_at;
            }
            (ToolCallBlock::Other(new), ToolCallBlock::Other(old)) => {
                new.started_at = old.started_at;
            }
            // Variant mismatch (shouldn't happen in practice) — skip.
            _ => {}
        }
    }

    /// Whether the tool call finished without an error.
    pub fn is_success(&self) -> bool {
        match self {
            ToolCallBlock::Execute(b) => b.is_success(),
            ToolCallBlock::Read(b) => b.is_success(),
            ToolCallBlock::Edit(b) => b.is_success(),
            ToolCallBlock::Search(b) => b.is_success(),
            ToolCallBlock::ListDir(b) => b.is_success(),
            ToolCallBlock::WebFetch(b) => b.is_success(),
            ToolCallBlock::WebSearch(b) => b.is_success(),
            ToolCallBlock::IntegrationSearch(b) => b.is_success(),
            ToolCallBlock::UseTool(b) => b.is_success(),
            ToolCallBlock::MemorySearch(b) => b.is_success(),
            ToolCallBlock::Skill(b) => b.is_success(),
            ToolCallBlock::Other(b) => b.is_success(),
            ToolCallBlock::Lifecycle(_) => true,
        }
    }

    /// Set `started_at` on the inner variant block.
    ///
    /// Unlike `transfer_timing_from`, this works across variant boundaries
    /// (e.g. setting `started_at` on a `Search` block from a value captured
    /// when the block was still `Other`).
    pub fn set_started_at(&mut self, instant: std::time::Instant) {
        match self {
            ToolCallBlock::Execute(b) => b.started_at = Some(instant),
            ToolCallBlock::Read(b) => b.started_at = Some(instant),
            ToolCallBlock::Edit(b) => b.started_at = Some(instant),
            ToolCallBlock::Search(b) => b.started_at = Some(instant),
            ToolCallBlock::ListDir(b) => b.started_at = Some(instant),
            ToolCallBlock::WebFetch(b) => b.started_at = Some(instant),
            ToolCallBlock::WebSearch(b) => b.started_at = Some(instant),
            ToolCallBlock::IntegrationSearch(b) => b.started_at = Some(instant),
            ToolCallBlock::UseTool(b) => b.started_at = Some(instant),
            ToolCallBlock::MemorySearch(b) => b.started_at = Some(instant),
            ToolCallBlock::Skill(b) => b.started_at = Some(instant),
            ToolCallBlock::Other(b) => b.started_at = Some(instant),
            // Lifecycle events have no timing.
            ToolCallBlock::Lifecycle(_) => {}
        }
    }

    /// Start timing for this block (sets `started_at = now`).
    ///
    /// Called when a block enters running UI state. Only blocks that
    /// actually run in the UI get meaningful timing. Pre-completed blocks
    /// keep `started_at = None` and show no timing data.
    pub fn start_timing(&mut self) {
        match self {
            ToolCallBlock::Execute(b) => {
                if b.started_at.is_none() {
                    b.started_at = Some(std::time::Instant::now());
                }
            }
            ToolCallBlock::Read(b) => {
                if b.started_at.is_none() {
                    b.started_at = Some(std::time::Instant::now());
                }
            }
            ToolCallBlock::Edit(b) => {
                if b.started_at.is_none() {
                    b.started_at = Some(std::time::Instant::now());
                }
            }
            ToolCallBlock::Search(b) => {
                if b.started_at.is_none() {
                    b.started_at = Some(std::time::Instant::now());
                }
            }
            ToolCallBlock::ListDir(b) => {
                if b.started_at.is_none() {
                    b.started_at = Some(std::time::Instant::now());
                }
            }
            ToolCallBlock::WebFetch(b) => {
                if b.started_at.is_none() {
                    b.started_at = Some(std::time::Instant::now());
                }
            }
            ToolCallBlock::WebSearch(b) => {
                if b.started_at.is_none() {
                    b.started_at = Some(std::time::Instant::now());
                }
            }
            ToolCallBlock::IntegrationSearch(b) => {
                if b.started_at.is_none() {
                    b.started_at = Some(std::time::Instant::now());
                }
            }
            ToolCallBlock::UseTool(b) => {
                if b.started_at.is_none() {
                    b.started_at = Some(std::time::Instant::now());
                }
            }
            ToolCallBlock::MemorySearch(b) => {
                if b.started_at.is_none() {
                    b.started_at = Some(std::time::Instant::now());
                }
            }
            ToolCallBlock::Skill(b) => {
                if b.started_at.is_none() {
                    b.started_at = Some(std::time::Instant::now());
                }
            }
            ToolCallBlock::Other(b) => {
                if b.started_at.is_none() {
                    b.started_at = Some(std::time::Instant::now());
                }
            }
            // Lifecycle events have no timing.
            ToolCallBlock::Lifecycle(_) => {}
        }
    }

    /// Create from tool name string (for parsing ACP tool calls).
    pub fn from_name(name: &str, summary: impl Into<String>) -> Self {
        match name.to_lowercase().as_str() {
            "run_terminal_command" | "run_terminal_cmd" | "bash" | "shell" | "execute" => {
                ToolCallBlock::Execute(ExecuteToolCallBlock::new(summary))
            }
            "read_file" | "read" => ToolCallBlock::Read(ReadToolCallBlock::new(summary)),
            "search_replace" | "edit" | "apply_patch" | "strreplace" => {
                ToolCallBlock::Edit(EditToolCallBlock::new(summary, Vec::new()))
            }
            "write" => ToolCallBlock::Edit(
                EditToolCallBlock::new(summary, Vec::new()).with_prefix("Creating "),
            ),
            "list_dir" | "ls" => ToolCallBlock::ListDir(ListDirToolCallBlock::new(summary)),
            "grep" | "search" | "glob" => {
                ToolCallBlock::Search(SearchToolCallBlock::new(summary.into()))
            }
            "web_fetch" | "fetch" => ToolCallBlock::WebFetch(WebFetchToolCallBlock::new(summary)),
            "web_search" => ToolCallBlock::WebSearch(WebSearchToolCallBlock::new(summary)),
            "search_tool" => {
                ToolCallBlock::IntegrationSearch(IntegrationSearchToolCallBlock::new(summary))
            }
            "use_tool" => ToolCallBlock::UseTool(UseToolCallBlock::new(summary)),
            "skill" => ToolCallBlock::Skill(OtherToolCallBlock::new("Skill", summary)),
            _ => ToolCallBlock::Other(OtherToolCallBlock::new(name, summary)),
        }
    }

    /// Full stored SOURCE text of this tool call for full-text scrollback
    /// search.
    ///
    /// Reads stored source fields and the `copy_text` accessors that read
    /// source data — never lays out (`output()` / word-wrap) or
    /// syntax-highlights — so indexing stays cheap.
    pub(crate) fn searchable_text(&self) -> Option<String> {
        match self {
            ToolCallBlock::Execute(b) => join_searchable([
                Some(b.command.clone()),
                b.description.clone(),
                b.output.clone(),
                b.error.clone(),
            ]),
            ToolCallBlock::Read(b) => {
                join_searchable([Some(b.path.clone()), b.content.clone(), b.error.clone()])
            }
            ToolCallBlock::Edit(b) => join_searchable([Some(b.copy_text()), b.error.clone()]),
            ToolCallBlock::ListDir(b) => join_searchable([
                Some(b.path.clone()),
                Some(b.output.clone()),
                b.error.clone(),
            ]),
            ToolCallBlock::Search(b) => {
                // Each file group contributes its path plus every matched line.
                let file_matches = join_searchable(b.file_matches.iter().flat_map(|fm| {
                    std::iter::once(Some(fm.path.clone()))
                        .chain(fm.matches.iter().map(|m| Some(m.content.clone())))
                }));
                let file_paths = join_searchable(b.file_paths.iter().cloned().map(Some));
                join_searchable([
                    Some(b.pattern.clone()),
                    b.meta.path.clone(),
                    b.meta.glob.clone(),
                    b.meta.file_type.clone(),
                    file_paths,
                    file_matches,
                    b.error.clone(),
                ])
            }
            ToolCallBlock::WebFetch(b) => {
                join_searchable([Some(b.url.clone()), b.output.clone(), b.error.clone()])
            }
            ToolCallBlock::WebSearch(b) => {
                let citations = join_searchable(b.citations.iter().cloned().map(Some));
                join_searchable([
                    Some(b.query.clone()),
                    b.content.clone(),
                    citations,
                    b.label.clone(),
                    b.error.clone(),
                ])
            }
            ToolCallBlock::IntegrationSearch(b) => {
                join_searchable([Some(b.copy_text()), b.content.clone(), b.error.clone()])
            }
            ToolCallBlock::UseTool(b) => join_searchable([Some(b.copy_text()), b.error.clone()]),
            ToolCallBlock::MemorySearch(b) => {
                // Flatten each result's source, path, and snippet.
                let results = join_searchable(b.results.iter().flat_map(|r| {
                    [
                        Some(r.source.clone()),
                        Some(r.path.clone()),
                        Some(r.snippet.clone()),
                    ]
                }));
                join_searchable([Some(b.query.clone()), results, b.error.clone()])
            }
            ToolCallBlock::Skill(b) | ToolCallBlock::Other(b) => join_searchable([
                Some(b.name.clone()),
                Some(b.summary.clone()),
                b.output.clone(),
                b.error.clone(),
            ]),
            ToolCallBlock::Lifecycle(b) => join_searchable([Some(b.name.clone())]),
        }
    }

    /// Verb-group kind; `None` renders standalone and splits verb-group runs
    /// (still dense-packs via `is_groupable`).
    pub fn verb_group_kind(&self) -> Option<VerbGroupKind> {
        match self {
            ToolCallBlock::Read(b) => Some(if b.is_skill_read() {
                VerbGroupKind::Skill
            } else {
                VerbGroupKind::File
            }),
            ToolCallBlock::ListDir(_) => Some(VerbGroupKind::Dir),
            ToolCallBlock::Search(_) => Some(VerbGroupKind::Search),
            ToolCallBlock::WebFetch(_) => Some(VerbGroupKind::WebFetch),
            ToolCallBlock::WebSearch(_) => Some(VerbGroupKind::WebSearch),
            ToolCallBlock::IntegrationSearch(_) => Some(VerbGroupKind::IntegrationSearch),
            ToolCallBlock::MemorySearch(_) => Some(VerbGroupKind::MemorySearch),
            ToolCallBlock::Skill(_) => Some(VerbGroupKind::Skill),
            ToolCallBlock::Execute(_)
            | ToolCallBlock::Edit(_)
            | ToolCallBlock::UseTool(_)
            | ToolCallBlock::Other(_)
            | ToolCallBlock::Lifecycle(_) => None,
        }
    }

    /// Bucket identity for aggregated header LABELS. Superset of
    /// [`Self::verb_group_kind`]: the action kinds excluded from eager verb
    /// folding still get a bucket when a truncation header describes the
    /// rows it hides. `None` only for lifecycle chrome, which is never
    /// worth labeling. Variants are listed explicitly so a new
    /// `ToolCallBlock` variant must decide here too.
    pub fn label_kind(&self) -> Option<VerbGroupKind> {
        match self {
            ToolCallBlock::Execute(_) => Some(VerbGroupKind::Command),
            ToolCallBlock::Edit(_) => Some(VerbGroupKind::EditFile),
            ToolCallBlock::UseTool(_) => Some(VerbGroupKind::McpCall),
            ToolCallBlock::Other(_) => Some(VerbGroupKind::OtherTool),
            ToolCallBlock::Lifecycle(_) => None,
            ToolCallBlock::Read(_)
            | ToolCallBlock::ListDir(_)
            | ToolCallBlock::Search(_)
            | ToolCallBlock::WebFetch(_)
            | ToolCallBlock::WebSearch(_)
            | ToolCallBlock::IntegrationSearch(_)
            | ToolCallBlock::MemorySearch(_)
            | ToolCallBlock::Skill(_) => self.verb_group_kind(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verb_is_tense_aware() {
        assert_eq!(VerbGroupKind::File.verb(false), "Read");
        assert_eq!(VerbGroupKind::File.verb(true), "Reading");
        assert_eq!(VerbGroupKind::Skill.verb(false), "Read");
        assert_eq!(VerbGroupKind::Search.verb(false), "Searched");
        assert_eq!(VerbGroupKind::Search.verb(true), "Searching");
        assert_eq!(VerbGroupKind::Dir.verb(false), "Listed");
        assert_eq!(VerbGroupKind::Dir.verb(true), "Listing");
        assert_eq!(VerbGroupKind::WebFetch.verb(false), "Fetched");
        assert_eq!(VerbGroupKind::WebFetch.verb(true), "Fetching");
        assert_eq!(VerbGroupKind::WebSearch.verb(false), "Searched");
        assert_eq!(VerbGroupKind::MemorySearch.verb(false), "Searched");
        assert_eq!(VerbGroupKind::IntegrationSearch.verb(true), "Searching");
        assert_eq!(VerbGroupKind::Subagent.verb(false), "Ran");
        assert_eq!(VerbGroupKind::Subagent.verb(true), "Running");
        assert_eq!(VerbGroupKind::Command.verb(false), "Ran");
        assert_eq!(VerbGroupKind::Command.verb(true), "Running");
        assert_eq!(VerbGroupKind::EditFile.verb(false), "Edited");
        assert_eq!(VerbGroupKind::EditFile.verb(true), "Editing");
        assert_eq!(VerbGroupKind::McpCall.verb(false), "Called");
        assert_eq!(VerbGroupKind::McpCall.verb(true), "Calling");
        assert_eq!(VerbGroupKind::OtherTool.verb(false), "Ran");
    }

    #[test]
    fn noun_pluralizes_by_count() {
        assert_eq!(VerbGroupKind::File.noun(1), "file");
        assert_eq!(VerbGroupKind::File.noun(2), "files");
        assert_eq!(VerbGroupKind::Skill.noun(2), "skills");
        assert_eq!(VerbGroupKind::Search.noun(1), "pattern");
        assert_eq!(VerbGroupKind::Dir.noun(2), "dirs");
        assert_eq!(VerbGroupKind::WebFetch.noun(1), "website");
        assert_eq!(VerbGroupKind::WebSearch.noun(2), "websites");
        // Irregular plural.
        assert_eq!(VerbGroupKind::MemorySearch.noun(1), "memory");
        assert_eq!(VerbGroupKind::MemorySearch.noun(2), "memories");
        assert_eq!(VerbGroupKind::IntegrationSearch.noun(1), "MCP tool");
        assert_eq!(VerbGroupKind::IntegrationSearch.noun(2), "MCP tools");
        assert_eq!(VerbGroupKind::Subagent.noun(1), "subagent");
        assert_eq!(VerbGroupKind::Subagent.noun(2), "subagents");
        assert_eq!(VerbGroupKind::Command.noun(1), "command");
        assert_eq!(VerbGroupKind::Command.noun(2), "commands");
        assert_eq!(VerbGroupKind::EditFile.noun(2), "files");
        assert_eq!(VerbGroupKind::McpCall.noun(1), "MCP tool");
        assert_eq!(VerbGroupKind::OtherTool.noun(1), "tool");
        assert_eq!(VerbGroupKind::OtherTool.noun(2), "tools");
    }

    #[test]
    fn every_variant_has_a_group_decision() {
        let blocks = [
            ToolCallBlock::Execute(ExecuteToolCallBlock::new("ls")),
            ToolCallBlock::Read(ReadToolCallBlock::new("src/main.rs")),
            ToolCallBlock::Read(ReadToolCallBlock::new("/x/skills/deploy/SKILL.md")),
            ToolCallBlock::Edit(EditToolCallBlock::new("src/main.rs", Vec::new())),
            ToolCallBlock::ListDir(ListDirToolCallBlock::new("src")),
            ToolCallBlock::Search(SearchToolCallBlock::new("todo")),
            ToolCallBlock::WebFetch(WebFetchToolCallBlock::new("https://example.com")),
            ToolCallBlock::WebSearch(WebSearchToolCallBlock::new("grok")),
            ToolCallBlock::IntegrationSearch(IntegrationSearchToolCallBlock::new("linear")),
            ToolCallBlock::UseTool(UseToolCallBlock::new("linear__save_issue")),
            ToolCallBlock::MemorySearch(MemorySearchToolCallBlock::new("auth")),
            ToolCallBlock::Skill(OtherToolCallBlock::new("Skill", "deploy")),
            ToolCallBlock::Other(OtherToolCallBlock::new("todo_write", "update")),
            ToolCallBlock::Lifecycle(LifecycleEventBlock::new("session_start")),
        ];
        for block in &blocks {
            // Exhaustive on purpose: a new variant fails compilation here
            // until it gets an explicit verb-grouping decision.
            let expected = match block {
                ToolCallBlock::Read(b) if b.is_skill_read() => Some(VerbGroupKind::Skill),
                ToolCallBlock::Read(_) => Some(VerbGroupKind::File),
                ToolCallBlock::ListDir(_) => Some(VerbGroupKind::Dir),
                ToolCallBlock::Search(_) => Some(VerbGroupKind::Search),
                ToolCallBlock::WebFetch(_) => Some(VerbGroupKind::WebFetch),
                ToolCallBlock::WebSearch(_) => Some(VerbGroupKind::WebSearch),
                ToolCallBlock::IntegrationSearch(_) => Some(VerbGroupKind::IntegrationSearch),
                ToolCallBlock::MemorySearch(_) => Some(VerbGroupKind::MemorySearch),
                ToolCallBlock::Skill(_) => Some(VerbGroupKind::Skill),
                ToolCallBlock::Execute(_)
                | ToolCallBlock::Edit(_)
                | ToolCallBlock::UseTool(_)
                | ToolCallBlock::Other(_)
                | ToolCallBlock::Lifecycle(_) => None,
            };
            assert_eq!(block.verb_group_kind(), expected, "block: {block:?}");
        }
    }

    #[test]
    fn label_kind_extends_verb_kinds_to_action_tools() {
        assert_eq!(
            ToolCallBlock::Execute(ExecuteToolCallBlock::new("ls")).label_kind(),
            Some(VerbGroupKind::Command)
        );
        assert_eq!(
            ToolCallBlock::Edit(EditToolCallBlock::new("src/main.rs", Vec::new())).label_kind(),
            Some(VerbGroupKind::EditFile)
        );
        assert_eq!(
            ToolCallBlock::UseTool(UseToolCallBlock::new("linear__save_issue")).label_kind(),
            Some(VerbGroupKind::McpCall)
        );
        assert_eq!(
            ToolCallBlock::Other(OtherToolCallBlock::new("todo_write", "update")).label_kind(),
            Some(VerbGroupKind::OtherTool)
        );
        assert_eq!(
            ToolCallBlock::Lifecycle(LifecycleEventBlock::new("session_start")).label_kind(),
            None
        );
        // Verb-groupable kinds defer to the fold's own classification.
        assert_eq!(
            ToolCallBlock::Read(ReadToolCallBlock::new("src/main.rs")).label_kind(),
            Some(VerbGroupKind::File)
        );
        assert_eq!(
            ToolCallBlock::Read(ReadToolCallBlock::new("/x/skills/deploy/SKILL.md")).label_kind(),
            Some(VerbGroupKind::Skill)
        );
    }

    #[test]
    fn render_output_preview_handles_none_and_empty() {
        let theme = Theme::current();
        let (lines, total) = render_output_preview(None, &theme, 3, PreviewStyle::Plain);
        assert!(lines.is_empty());
        assert_eq!(total, 0);
        let (lines, total) = render_output_preview(Some(""), &theme, 3, PreviewStyle::Plain);
        assert!(lines.is_empty());
        assert_eq!(total, 0);
        let (lines, total) = render_output_preview(Some("a\nb\nc"), &theme, 0, PreviewStyle::Plain);
        assert!(lines.is_empty());
        assert_eq!(total, 0);
    }

    #[test]
    fn render_output_preview_truncates_to_max() {
        let theme = Theme::current();
        let (lines, total) =
            render_output_preview(Some("a\nb\nc\nd\ne"), &theme, 3, PreviewStyle::Plain);
        assert_eq!(lines.len(), 3, "only the first max_lines are rendered");
        assert_eq!(total, 5, "total reflects the full source line count");
    }

    #[test]
    fn render_output_preview_under_max_returns_all() {
        let theme = Theme::current();
        let (lines, total) = render_output_preview(Some("a\nb"), &theme, 3, PreviewStyle::Plain);
        assert_eq!(lines.len(), 2);
        assert_eq!(total, 2);
    }

    #[test]
    fn preview_hint_line_only_when_truncated() {
        let theme = Theme::current();
        assert!(preview_hint_line(&theme, 3, 2).is_none());
        assert!(preview_hint_line(&theme, 3, 0).is_none());
        let hint = preview_hint_line(&theme, 3, 5).expect("hint when total > shown");
        let plain: String = hint
            .content
            .spans
            .iter()
            .map(|s| s.content.as_ref())
            .collect();
        assert!(plain.contains("2 more lines"), "hint text: {plain}");
    }

    #[test]
    fn append_collapsed_body_shows_error_reason() {
        let theme = Theme::current();
        let mut lines = Vec::new();
        append_collapsed_body(
            &mut lines,
            &theme,
            Some("exit code 1: permission denied"),
            None,
            3,
            PreviewStyle::Plain,
        );
        assert!(
            !lines.is_empty(),
            "failed tools must surface an error body when collapsed"
        );
        let plain: String = lines
            .iter()
            .flat_map(|l| l.content.spans.iter().map(|s| s.content.as_ref()))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            plain.contains("permission denied"),
            "error text must be visible, got: {plain}"
        );
        assert!(
            plain.contains('✗'),
            "error lines should carry a failure marker"
        );
    }

    #[test]
    fn append_collapsed_body_error_shown_even_when_preview_disabled() {
        let theme = Theme::current();
        let mut lines = Vec::new();
        // max_output_lines = 0 disables the stdout preview, but the error
        // reason must still appear — that is why the block is red.
        append_collapsed_body(
            &mut lines,
            &theme,
            Some("command not found"),
            Some("stdout that should not appear"),
            0,
            PreviewStyle::Plain,
        );
        let plain: String = lines
            .iter()
            .flat_map(|l| l.content.spans.iter().map(|s| s.content.as_ref()))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(plain.contains("command not found"));
        assert!(
            !plain.contains("stdout that should not appear"),
            "output must stay hidden when max_output_lines is 0"
        );
    }

    #[test]
    fn terminal_preview_strips_escape_codes_and_keeps_text() {
        let theme = Theme::current();
        // ANSI-wrapped "fail" then a second line "ok". The terminal preview
        // must feed the stream through the VTE emulator so escape codes are
        // not painted as literal text.
        let raw = "\x1b[31mfail\x1b[0m\nok";
        let (lines, total) = render_output_preview(Some(raw), &theme, 3, PreviewStyle::Terminal);
        assert_eq!(total, 2);
        assert_eq!(lines.len(), 2);
        let plain: String = lines
            .iter()
            .flat_map(|l| l.content.spans.iter().map(|s| s.content.as_ref()))
            .collect::<Vec<_>>()
            .join("|");
        assert!(
            plain.contains("fail") && plain.contains("ok"),
            "de-escaped text must be present, got: {plain}"
        );
        assert!(
            !plain.contains("\x1b") && !plain.contains("[31m"),
            "raw escape sequences must not appear, got: {plain}"
        );
    }
}
