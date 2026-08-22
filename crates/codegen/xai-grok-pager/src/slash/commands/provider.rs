//! `/provider` — open the provider management dialog.
//!
//! The single surface for provider work: status, connect (masked key
//! modal), disconnect, and live catalog refresh, all inside one dialog.
//! Also reachable from the command palette (Ctrl+P).

use crate::app::actions::Action;
use crate::slash::command::{CommandExecCtx, CommandResult, SlashCommand};

pub struct ProviderCommand;

impl SlashCommand for ProviderCommand {
    fn name(&self) -> &str {
        "provider"
    }

    fn aliases(&self) -> &[&str] {
        &["providers"]
    }

    fn description(&self) -> &str {
        "Manage providers: connect, disconnect, refresh model catalogs"
    }

    fn usage(&self) -> &str {
        "/provider"
    }

    fn takes_args(&self) -> bool {
        false
    }

    fn run(&self, _ctx: &mut CommandExecCtx, args: &str) -> CommandResult {
        if !args.trim().is_empty() {
            return CommandResult::Error(
                "/provider takes no arguments — use the dialog to pick a \
                 provider and press c to connect, d to disconnect, r to \
                 refresh."
                    .into(),
            );
        }
        CommandResult::Action(Action::OpenProviders)
    }
}
