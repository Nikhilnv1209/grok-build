//! Secret-input modal for `/connect <provider>`.
//!
//! Mirrors the loopback auth-token box (masked input + paste) but hosted in
//! the shared modal system, so a provider API key is collected inside the TUI
//! without ever landing in scrollback. `SubmitConnectKey` persists the key to
//! `~/.grok/auth.json` and triggers a live catalog reload, so connected
//! models become selectable immediately.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};

use crate::app::actions::Action;
use crate::input::line_editor::{LineEditOutcome, LineEditor};
use crate::theme::Theme;
use crate::views::modal_window::{self, ModalSizing, ModalWindowConfig};
use xai_grok_shell::providers::ProviderSpec;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectSubmit {
    pub provider: String,
    pub key: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectKeyOutcome {
    Changed,
    Submit(ConnectSubmit),
    Close,
    Unhandled,
}

pub struct ConnectProviderModal {
    pub provider: ProviderSpec,
    secret: LineEditor,
    pub window: modal_window::ModalWindowState,
}

impl ConnectProviderModal {
    pub fn open(provider: Option<&ProviderSpec>) -> Option<Self> {
        let provider = provider.cloned()?;
        Some(Self {
            provider,
            secret: LineEditor::default(),
            window: modal_window::ModalWindowState::default(),
        })
    }

    pub fn handle_key(&mut self, key: &KeyEvent) -> ConnectKeyOutcome {
        match key.code {
            KeyCode::Enter => {
                let key = self.secret.text().trim();
                if key.is_empty() {
                    return ConnectKeyOutcome::Changed;
                }
                ConnectKeyOutcome::Submit(ConnectSubmit {
                    provider: self.provider.id.to_string(),
                    key: key.to_string(),
                })
            }
            KeyCode::Esc => ConnectKeyOutcome::Close,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                ConnectKeyOutcome::Close
            }
            _ => {
                // Reuse the prompt's single-line editor semantics (sanitize,
                // cursor, backspace) by feeding single characters through the
                // paste path.
                let mut outcome = LineEditOutcome::Unhandled;
                if let KeyCode::Char(c) = key.code
                    && !key.modifiers.contains(KeyModifiers::SUPER)
                {
                    let _ = self.secret.insert_paste(&c.to_string());
                    outcome = LineEditOutcome::TextChanged;
                }
                let _ = outcome;
                ConnectKeyOutcome::Changed
            }
        }
    }

    pub fn handle_paste(&mut self, text: &str) -> ConnectKeyOutcome {
        let _ = self.secret.insert_paste(text);
        ConnectKeyOutcome::Changed
    }

    pub fn cursor_byte(&self) -> usize {
        self.secret.cursor_byte()
    }

    pub fn secret_len(&self) -> usize {
        self.secret.text().chars().count()
    }
}

pub fn render(
    modal: &ConnectProviderModal,
    buf: &mut Buffer,
    area: Rect,
    theme: &Theme,
    compact: bool,
) {
    // The dialog is a small form (a few info rows + one input line), so
    // pre-shrink the draw area to its natural height and let the chrome
    // center it instead of expanding to most of the terminal.
    let area = fit_area(area);
    let title = format!("Connect {}", modal.provider.name);
    let modal_config = ModalWindowConfig {
        title: &title,
        tabs: None,
        shortcuts: &[],
        sizing: ModalSizing {
            width_pct: 0.5,
            max_width: 64,
            min_width: 44,
            v_margin: 0,
            h_pad: 2,
            v_pad: 1,
            footer_lines: 1,
        }
        .with_compact(compact),
        fold_info: None,
    };

    // `render_modal_window` needs a `&mut` window; the static render path
    // draws from a snapshot clone because connect is a small, stateless body.
    let mut window = modal.window.clone();
    let Some(meta) = modal_window::render_modal_window(buf, area, &mut window, &modal_config, theme)
    else {
        return;
    };
    let content = meta.content;
    if content.height < 6 || content.width < 20 {
        return;
    }

    let mut y = content.y;
    for line in [
        format!("Provider: {}", modal.provider.name),
        format!("Env var: {}", modal.provider.env_keys.join(" or ")),
        format!("Docs: {}", modal.provider.docs_url),
        String::new(),
    ] {
        if y >= content.y + content.height - 2 {
            break;
        }
        let (truncated, _) =
            truncate_to_width(&line, content.width.saturating_sub(4) as usize);
        for (col, ch) in truncated.chars().enumerate() {
            if let Some(cell) = buf.cell_mut((content.x + col as u16, y)) {
                cell.set_symbol(&ch.to_string());
                cell.set_style(Style::default().fg(theme.text_primary));
            }
        }
        y += 1;
    }

    let input_y = content.y + content.height - 2;
    let placeholder = if modal.secret_len() == 0 {
        "Paste or type your API key, then Enter — Esc to cancel"
    } else {
        ""
    };
    let (masked, cursor_col) = masked_view(
        &placeholder.to_string(),
        modal.secret_len(),
        content.width.saturating_sub(2) as usize,
    );
    let input_style = Style::default()
        .fg(theme.accent_user)
        .add_modifier(Modifier::BOLD);
    for (col, ch) in masked.chars().enumerate() {
        if let Some(cell) = buf.cell_mut((content.x + col as u16, input_y)) {
            cell.set_symbol(&ch.to_string());
            cell.set_style(input_style);
        }
    }
    if let Some(cell) = buf.cell_mut((content.x + cursor_col as u16, input_y)) {
        cell.set_style(Style::default().fg(theme.bg_base).bg(theme.text_primary));
    }
    let _ = Action::CancelConnectKey;
}

/// Natural height of the dialog: borders + v_pad + footer + 4 info rows +
/// a masked input row. Clamped so short terminals can still construct it.
fn fit_area(area: Rect) -> Rect {
    let height = area.height.min(10);
    Rect {
        x: area.x,
        y: area.y + (area.height.saturating_sub(height)) / 2,
        width: area.width,
        height,
    }
}

/// Render the secret fully masked; the placeholder shows only while empty.
fn masked_view(placeholder: &str, secret_len: usize, width: usize) -> (String, usize) {
    if secret_len == 0 {
        let (text, _) = truncate_to_width(placeholder, width);
        return (text, 0);
    }
    let dots = width.min(secret_len);
    ("•".repeat(dots), dots)
}

fn truncate_to_width(text: &str, width: usize) -> (String, usize) {
    let mut count = 0usize;
    let mut out = String::new();
    for ch in text.chars() {
        if count >= width {
            break;
        }
        out.push(ch);
        count += 1;
    }
    (out, count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use xai_grok_shell::providers::find_provider;

    fn key(code: KeyCode, modifiers: KeyModifiers) -> KeyEvent {
        KeyEvent::new(code, modifiers)
    }

    #[test]
    fn empty_enter_does_not_submit() {
        let mut modal = ConnectProviderModal::open(find_provider("opencode")).unwrap();
        assert!(matches!(
            modal.handle_key(&key(KeyCode::Enter, KeyModifiers::NONE)),
            ConnectKeyOutcome::Changed
        ));
    }

    #[test]
    fn typed_chars_then_enter_submits_secret() {
        let mut modal = ConnectProviderModal::open(find_provider("opencode")).unwrap();
        for c in ['s', 'k', '-', 'x'] {
            assert!(matches!(
                modal.handle_key(&key(KeyCode::Char(c), KeyModifiers::NONE)),
                ConnectKeyOutcome::Changed
            ));
        }
        match modal.handle_key(&key(KeyCode::Enter, KeyModifiers::NONE)) {
            ConnectKeyOutcome::Submit(submit) => {
                assert_eq!(submit.provider, "opencode");
                assert_eq!(submit.key, "sk-x");
            }
            other => panic!("expected Submit, got {other:?}"),
        }
    }

    #[test]
    fn escape_cancels_and_paste_accumulates() {
        let mut modal = ConnectProviderModal::open(find_provider("deepseek")).unwrap();
        assert!(matches!(modal.handle_paste("sk-abc"), ConnectKeyOutcome::Changed));
        assert!(matches!(
            modal.handle_key(&key(KeyCode::Esc, KeyModifiers::NONE)),
            ConnectKeyOutcome::Close
        ));
        assert_eq!(modal.secret_len(), 6, "paste text is retained");
    }

    #[test]
    fn fit_area_caps_height_and_centers() {
        let area = Rect::new(0, 0, 120, 40);
        let fitted = fit_area(area);
        assert_eq!(fitted.width, 120);
        assert_eq!(fitted.x, 0);
        assert_eq!(fitted.height, 10);
        assert_eq!(fitted.y, 15, "must stay vertically centered");

        let tiny = Rect::new(4, 2, 30, 6);
        let fitted = fit_area(tiny);
        assert_eq!(fitted, tiny, "already-small areas are unchanged");
    }
}