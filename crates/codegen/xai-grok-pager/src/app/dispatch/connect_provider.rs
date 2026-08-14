//! Dispatch for the `/connect` secret-input modal and its submit resolution.

use crate::app::actions::Effect;
use crate::app::app_view::AppView;
use crate::app::dispatch::ctx::with_active_agent;
use crate::views::connect_provider_modal::ConnectProviderModal;
use crate::views::modal::ActiveModal;
use xai_grok_shell::providers;

/// Open the masked API-key modal for a provider on the visible agent.
/// Falls back to a toast when no agent view is active (welcome screen).
pub(super) fn dispatch_connect_provider(app: &mut AppView, provider_id: &str) -> Vec<Effect> {
    let Some(modal_state) = ConnectProviderModal::open(providers::find_provider(provider_id)) else {
        app.show_toast("Unknown provider");
        return vec![];
    };
    let mut opened = false;
    with_active_agent(app, |agent| {
        agent.active_modal = Some(ActiveModal::ConnectProvider {
            state: Box::new(modal_state),
        });
        opened = true;
    });
    if !opened {
        app.show_toast("Start a session first, then run /connect.");
    }
    vec![]
}

/// Persist the submitted key, then ask the agent to re-resolve its catalog
/// (`x.ai/internal/reload_models`) so connected models become selectable
/// without a restart; the refreshed list arrives as `x.ai/models/update`.
pub(super) fn dispatch_submit_connect_key(
    app: &mut AppView,
    provider_id: &str,
    key: &str,
) -> Vec<Effect> {
    let mut effects = Vec::new();
    let message = match providers::store_key(provider_id, key) {
        Ok(()) => {
            let name = providers::find_provider(provider_id)
                .map(|s| s.name)
                .unwrap_or(provider_id);
            // The embedded agent (default TUI mode) has no config watcher,
            // so ask the agent to re-resolve its catalog right away; the
            // refreshed list arrives as `x.ai/models/update`. Leader mode
            // dedupes with its own auth.json watcher.
            effects.push(Effect::ReloadAgentModels);
            format!("Connected {name}. Models refresh automatically.")
        }
        Err(error) => format!("Connect failed: {error}"),
    };
    app.show_toast(&message);
    effects
}

/// Drop a provider's stored key and ask the agent to re-resolve its catalog
/// so the removed models disappear without a restart (mirrors connect).
pub(super) fn dispatch_disconnect_provider(app: &mut AppView, provider_id: &str) -> Vec<Effect> {
    let spec = providers::find_provider(provider_id);
    let message = match spec {
        None => format!("Unknown provider '{provider_id}'."),
        Some(spec) => match providers::clear_key(spec.id) {
            Ok(true) => format!(
                "Disconnected {}. Environment variables are left unchanged.",
                spec.name
            ),
            Ok(false) => {
                if providers::is_connected(spec) {
                    format!(
                        "No stored API key for {}. {}",
                        spec.name,
                        format!("{} is still set in the environment.", spec.env_keys.join(" or "))
                    )
                } else {
                    format!("No stored API key for {}. Nothing to remove.", spec.name)
                }
            }
            Err(error) => format!("Failed to disconnect: {error}"),
        },
    };
    // The reload is idempotent; fire it for every known provider so a stale
    // catalog can never outlive a disconnect attempt.
    let effects = spec
        .is_some()
        .then_some(Effect::ReloadAgentModels)
        .into_iter()
        .collect();
    app.show_toast(&message);
    effects
}