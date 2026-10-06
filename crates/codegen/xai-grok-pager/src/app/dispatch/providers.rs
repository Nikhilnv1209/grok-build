//! Dispatch for the `/provider` management dialog.
//!
//! The dialog is the single surface for provider work: connect hands off to
//! the masked key modal, disconnect drops the stored key. Every mutating path
//! funnels through the existing connect/disconnect dispatchers so the
//! model-reload effect stays in one place. Provider model metadata is
//! checked in, so there is no catalog refresh to trigger.

use crate::app::actions::Effect;
use crate::app::app_view::AppView;
use crate::app::dispatch::ctx::with_active_agent;
use crate::views::modal::ActiveModal;
use crate::views::providers_modal::ProvidersModal;

fn open_dialog(agent_modal: &mut Option<ActiveModal>) {
    *agent_modal = Some(ActiveModal::Providers {
        state: Box::new(ProvidersModal::open()),
    });
}

/// Open the management dialog on the visible agent; before any session
/// exists (fresh launch), host it at app level so `/provider` still works.
pub(super) fn dispatch_open_providers(app: &mut AppView) -> Vec<Effect> {
    let mut opened = false;
    with_active_agent(app, |agent| {
        open_dialog(&mut agent.active_modal);
        opened = true;
    });
    if !opened {
        app.welcome_modal = Some(ActiveModal::Providers {
            state: Box::new(ProvidersModal::open()),
        });
    }
    vec![]
}

/// Re-show the dialog after the key-entry modal closes, so the flow returns
/// to the list and the fresh connection state is visible immediately.
pub(super) fn reopen_providers_dialog(app: &mut AppView) {
    with_active_agent(app, |agent| {
        open_dialog(&mut agent.active_modal);
    });
}
