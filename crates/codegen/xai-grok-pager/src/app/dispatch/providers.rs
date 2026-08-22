//! Dispatch for the `/provider` management dialog and live catalog refresh.
//!
//! The dialog is the single surface for provider work: connect hands off to
//! the masked key modal, disconnect drops the stored key, refresh refetches
//! live catalogs. Every mutating path funnels through the existing connect/
//! disconnect dispatchers so the model-reload effect stays in one place.

use crate::app::actions::Effect;
use crate::app::app_view::AppView;
use crate::app::dispatch::ctx::with_active_agent;
use crate::views::modal::ActiveModal;
use crate::views::providers_modal::ProvidersModal;

/// Guard so startup refresh fires exactly once per app run no matter how
/// many sessions are created or resumed.
static AUTO_REFRESH_DONE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Kick one background live-catalog refresh after the first session becomes
/// ready, so provider models published since the last run show up without a
/// restart (the completion task result reloads the agent catalog again).
pub(super) fn auto_refresh_once_per_run() -> Vec<Effect> {
    use std::sync::atomic::Ordering;
    if AUTO_REFRESH_DONE.swap(true, Ordering::Relaxed) {
        return vec![];
    }
    vec![Effect::RefreshProviderCatalogs { provider: None }]
}

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

/// Kick off a background live-catalog refresh (`None` = every connected
/// provider). Marks rows in-flight; `TaskResult::ProviderCatalogsRefreshed`
/// clears them, toasts the summary, and reloads the agent catalog when any
/// fetch succeeded.
pub(super) fn dispatch_refresh_provider_models(
    app: &mut AppView,
    provider: Option<String>,
) -> Vec<Effect> {
    let id = provider.clone();
    with_active_agent(app, |agent| {
        if let Some(ActiveModal::Providers { state }) = agent.active_modal.as_mut() {
            state.set_refreshing(id.as_deref(), true);
        }
    });
    vec![Effect::RefreshProviderCatalogs { provider }]
}
