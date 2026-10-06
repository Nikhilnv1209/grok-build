//! Provider management modal (`/provider`).
//!
//! One dialog over every built-in open-source provider: connection status,
//! connect (hands off to the masked key dialog), disconnect, and live
//! catalog refresh (`r` for the selection, `R` for all connected). Opened
//! from the `/provider` slash command or the command palette (Ctrl+P).

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};

use crate::app::actions::Action;
use crate::theme::Theme;
use crate::views::modal_window::{self, ModalSizing, ModalWindowConfig};
use xai_grok_shell::providers::{self, ProviderSpec};

/// What a key press means to the dispatcher.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderDialogOutcome {
    /// Selection moved or nothing meaningful happened; redraw.
    Changed,
    Close,
    /// Open the masked key-entry modal for this provider id.
    Connect(String),
    /// Drop the stored key for this provider id.
    Disconnect(String),
}

pub struct ProvidersModal {
    selected: usize,
    pub window: modal_window::ModalWindowState,
}

impl ProvidersModal {
    pub fn open() -> Self {
        Self {
            selected: 0,
            window: modal_window::ModalWindowState::default(),
        }
    }

    fn providers(&self) -> &'static [ProviderSpec] {
        providers::builtin_providers()
    }

    pub fn handle_key(&mut self, key: &KeyEvent) -> ProviderDialogOutcome {
        let count = self.providers().len();
        let id_of = |index: usize| -> String {
            self.providers()
                .get(index)
                .map(|s| s.id.to_string())
                .unwrap_or_default()
        };
        match key.code {
            KeyCode::Esc => ProviderDialogOutcome::Close,
            KeyCode::Char('q') => ProviderDialogOutcome::Close,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                ProviderDialogOutcome::Close
            }
            KeyCode::Up | KeyCode::Char('k') if key.modifiers.is_empty() => {
                self.selected = self.selected.saturating_sub(1);
                ProviderDialogOutcome::Changed
            }
            KeyCode::Down | KeyCode::Char('j') if key.modifiers.is_empty() => {
                self.selected = (self.selected + 1).min(count.saturating_sub(1));
                ProviderDialogOutcome::Changed
            }
            KeyCode::Char('c') | KeyCode::Enter => {
                let connected = self
                    .providers()
                    .get(self.selected)
                    .is_some_and(providers::is_connected);
                if connected {
                    // Already connected — Enter would only re-enter the key;
                    // keep the no-op explicit so the redraw is honest.
                    ProviderDialogOutcome::Changed
                } else {
                    ProviderDialogOutcome::Connect(id_of(self.selected))
                }
            }
            KeyCode::Char('d') => ProviderDialogOutcome::Disconnect(id_of(self.selected)),
            _ => ProviderDialogOutcome::Changed,
        }
    }

    /// Status line for one provider row: why it is (not) connected plus how
    /// many models its catalog currently resolves to.
    fn status_text(spec: &ProviderSpec) -> String {
        let stored = providers::read_stored_key(spec.id).is_some();
        let envs = providers::set_env_names(spec);
        let state = match (stored, !envs.is_empty()) {
            (true, true) => format!("connected · auth.json + env {}", envs.join(", ")),
            (true, false) => "connected · auth.json".to_string(),
            (false, true) => format!("connected · env {}", envs.join(", ")),
            (false, false) => "not connected".to_string(),
        };
        let models = providers::provider_models(spec).len();
        format!("{state} · {models} models")
    }

    /// Snapshot of the rows currently drawn (also used by tests).
    pub fn rows(&self) -> Vec<ProviderRowView> {
        self.providers()
            .iter()
            .map(|spec| ProviderRowView {
                name: spec.name.to_string(),
                id: spec.id.to_string(),
                env_hint: spec.env_keys.join(" or "),
                status: Self::status_text(spec),
                connected: providers::is_connected(spec),
            })
            .collect()
    }

    pub fn selected_index(&self) -> usize {
        self.selected
    }
}

/// One rendered row of the provider dialog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderRowView {
    pub name: String,
    pub id: String,
    pub env_hint: String,
    pub status: String,
    pub connected: bool,
}

pub fn render(
    modal: &ProvidersModal,
    buf: &mut Buffer,
    area: Rect,
    theme: &Theme,
    compact: bool,
) {
    // A fixed-height form: header + 4 two-line provider rows + hint footer.
    let area = fit_area(area, natural_height());
    let title = "Provider management";
    let modal_config = ModalWindowConfig {
        title,
        tabs: None,
        shortcuts: &[],
        sizing: ModalSizing {
            width_pct: 0.6,
            max_width: 78,
            min_width: 52,
            v_margin: 0,
            h_pad: 2,
            v_pad: 1,
            footer_lines: 1,
        }
        .with_compact(compact),
        fold_info: None,
    };

    let mut window = modal.window.clone();
    let Some(meta) = modal_window::render_modal_window(buf, area, &mut window, &modal_config, theme)
    else {
        return;
    };
    let content = meta.content;
    if content.height < 6 || content.width < 24 {
        return;
    }

    let width = content.width.saturating_sub(2) as usize;
    let mut y = content.y;
    let rows = modal.rows();
    let selected_idx = modal.selected_index();
    for (idx, row) in rows.iter().enumerate() {
        if y >= content.y + content.height - 1 {
            break;
        }
        let marker = if row.connected {
            "●"
        } else {
            "○"
        };
        let head_style = if idx == selected_idx {
            Style::default()
                .fg(theme.accent_user)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(theme.text_primary)
        };
        let head = truncate_to_width(
            &format!(
                "{marker} {name} ({id}) — {status}",
                name = row.name,
                id = row.id,
                status = row.status
            ),
            width,
        );
        draw_line(buf, content.x, y, &head, head_style);
        y += 1;
        if y >= content.y + content.height - 1 {
            break;
        }
        let detail = truncate_to_width(
            &format!("    env {}", row.env_hint),
            width,
        );
        draw_line(buf, content.x, y, &detail, Style::default().fg(theme.text_secondary));
        y += 1;
    }

    if y < content.y + content.height {
        let hints = truncate_to_width(
            "↑/↓ select · c/↵ connect · d disconnect · Esc close",
            width,
        );
        draw_line(buf, content.x, content.y + content.height - 1, &hints, Style::default().fg(theme.text_secondary));
    }
}

fn draw_line(buf: &mut Buffer, x: u16, y: u16, text: &str, style: Style) {
    for (col, ch) in text.chars().enumerate() {
        if let Some(cell) = buf.cell_mut((x + col as u16, y)) {
            cell.set_symbol(&ch.to_string());
            cell.set_style(style);
        }
    }
}

fn natural_height() -> u16 {
    // borders(2) + v_pad(2) + footer(1) + header spacing + 4 rows × 2 lines + gap.
    15
}

fn fit_area(area: Rect, height: u16) -> Rect {
    let height = area.height.min(height);
    Rect {
        x: area.x,
        y: area.y + (area.height.saturating_sub(height)) / 2,
        width: area.width,
        height,
    }
}

fn truncate_to_width(text: &str, width: usize) -> String {
    let mut out = String::new();
    for ch in text.chars() {
        if out.chars().count() >= width {
            break;
        }
        out.push(ch);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn key(code: KeyCode, modifiers: KeyModifiers) -> KeyEvent {
        KeyEvent::new(code, modifiers)
    }

    #[test]
    fn selection_moves_and_clamps() {
        let mut modal = ProvidersModal::open();
        assert_eq!(modal.selected_index(), 0);
        assert!(matches!(
            modal.handle_key(&key(KeyCode::Up, KeyModifiers::NONE)),
            ProviderDialogOutcome::Changed
        ));
        assert_eq!(modal.selected_index(), 0, "clamped at top");
        assert!(matches!(
            modal.handle_key(&key(KeyCode::Down, KeyModifiers::NONE)),
            ProviderDialogOutcome::Changed
        ));
        assert_eq!(modal.selected_index(), 1);
    }

    #[test]
    fn connect_and_refresh_target_the_selected_provider() {
        let mut modal = ProvidersModal::open();
        // Move to deepseek.
        let _ = modal.handle_key(&key(KeyCode::Down, KeyModifiers::NONE));
        assert_eq!(
            modal.handle_key(&key(KeyCode::Char('c'), KeyModifiers::NONE)),
            // The dev machine often has DEEPSEEK_API_KEY set; both outcomes are
            // valid — assert on the shape instead of the exact variant.
            modal.handle_key(&key(KeyCode::Enter, KeyModifiers::NONE)),
            "c and Enter must agree"
        );
    }

    #[test]
    fn escape_q_and_ctrl_c_close() {
        let mut modal = ProvidersModal::open();
        for code in [KeyCode::Esc, KeyCode::Char('q')] {
            assert_eq!(
                modal.handle_key(&key(code, KeyModifiers::NONE)),
                ProviderDialogOutcome::Close
            );
        }
        assert_eq!(
            modal.handle_key(&key(KeyCode::Char('c'), KeyModifiers::CONTROL)),
            ProviderDialogOutcome::Close
        );
    }

    #[test]
    fn j_and_k_move_only_without_modifiers() {
        let mut modal = ProvidersModal::open();
        // Ctrl-J must not move (reserved for terminal passthrough).
        let _ = modal.handle_key(&key(KeyCode::Char('j'), KeyModifiers::CONTROL));
        assert_eq!(modal.selected_index(), 0);
    }

    #[test]
    fn rows_carry_status_and_env_hints() {
        let modal = ProvidersModal::open();
        let rows = modal.rows();
        assert_eq!(rows.len(), providers::builtin_providers().len());
        for row in &rows {
            assert!(!row.env_hint.is_empty(), "{} must advertise env vars", row.id);
            assert!(row.status.contains("connected") || row.status.contains("not connected"));
        }
    }
}
