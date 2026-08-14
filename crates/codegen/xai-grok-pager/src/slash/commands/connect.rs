//! `/connect` — list or connect an open-source provider.
//! `/disconnect` — drop a stored provider API key.

use xai_grok_shell::providers::{self, ProviderSpec};

use crate::app::actions::Action;
use crate::slash::command::{AppCtx, ArgItem, CommandExecCtx, CommandResult, SlashCommand};

pub struct ConnectCommand;

impl SlashCommand for ConnectCommand {
    fn name(&self) -> &str {
        "connect"
    }

    fn description(&self) -> &str {
        "List open-source providers (Umans, DeepSeek, OpenCode, Command Code)"
    }

    fn usage(&self) -> &str {
        "/connect [provider]"
    }

    fn takes_args(&self) -> bool {
        true
    }

    fn arg_placeholder(&self) -> Option<&str> {
        Some("[provider]")
    }

    fn suggest_args(&self, _ctx: &AppCtx, _args_query: &str) -> Option<Vec<ArgItem>> {
        Some(provider_arg_items())
    }

    fn run(&self, ctx: &mut CommandExecCtx, args: &str) -> CommandResult {
        let trimmed = args.trim();
        if trimmed.is_empty() {
            return CommandResult::Message(status_message());
        }
        let Some(spec) = providers::find_provider(trimmed) else {
            return CommandResult::Error(unknown_provider(trimmed));
        };
        if providers::is_connected(spec) {
            return CommandResult::Message(connected_message(spec));
        }
        if ctx.session_id.is_some() {
            // Open the masked key-entry modal inside the TUI.
            return CommandResult::Action(Action::ConnectProvider(spec.id.to_string()));
        }
        CommandResult::Message(connect_instructions(spec))
    }
}

pub struct DisconnectCommand;

impl SlashCommand for DisconnectCommand {
    fn name(&self) -> &str {
        "disconnect"
    }

    fn description(&self) -> &str {
        "Remove a stored provider API key"
    }

    fn usage(&self) -> &str {
        "/disconnect <provider>"
    }

    fn takes_args(&self) -> bool {
        true
    }

    fn args_required(&self) -> bool {
        true
    }

    fn arg_placeholder(&self) -> Option<&str> {
        Some("<provider>")
    }

    fn suggest_args(&self, _ctx: &AppCtx, _args_query: &str) -> Option<Vec<ArgItem>> {
        // Only configured providers are disconnectable; hide the rest so the
        // dropdown can't offer a no-op.
        let items: Vec<ArgItem> = provider_arg_items()
            .into_iter()
            .filter(|item| {
                providers::find_provider(&item.display)
                    .is_some_and(providers::is_connected)
            })
            .collect();
        (!items.is_empty()).then_some(items)
    }

    fn run(&self, _ctx: &mut CommandExecCtx, args: &str) -> CommandResult {
        let trimmed = args.trim();
        if trimmed.is_empty() {
            return CommandResult::Error("Usage: /disconnect <provider>".into());
        }
        let Some(spec) = providers::find_provider(trimmed) else {
            return CommandResult::Error(unknown_provider(trimmed));
        };
        // Clearing the key, the toast, and the catalog re-resolve all live in
        // dispatch so the disconnect path can emit the model-reload effect.
        CommandResult::Action(Action::DisconnectProvider(spec.id.to_string()))
    }
}

fn provider_arg_items() -> Vec<ArgItem> {
    providers::builtin_providers()
        .iter()
        .map(|spec| ArgItem {
            display: spec.id.to_string(),
            match_text: format!("{} {}", spec.id, spec.name),
            insert_text: spec.id.to_string(),
            description: spec.name.to_string(),
        })
        .collect()
}

fn status_message() -> String {
    let mut out = String::from("Open-source providers\n");
    for spec in providers::builtin_providers() {
        let envs = providers::set_env_names(spec);
        let has_stored = providers::read_stored_key(spec.id).is_some();
        let state = match (has_stored, !envs.is_empty()) {
            (true, true) => "connected (auth.json + env)".to_string(),
            (true, false) => "connected (auth.json)".to_string(),
            (false, true) => format!("connected (env: {})", envs.join(", ")),
            (false, false) => "not connected".to_string(),
        };
        out.push_str(&format!(
            "\n  {id}  {name} — {state}\n    grok connect {id}   env {env}",
            id = spec.id,
            name = spec.name,
            state = state,
            env = spec.env_keys.join(" or "),
        ));
    }
    out.push_str("\n\nAfter connecting, pick a model with /model.");
    out
}

fn connected_message(spec: &ProviderSpec) -> String {
    let models = providers::provider_models(spec);
    let envs = providers::set_env_names(spec);
    let stored = providers::read_stored_key(spec.id).is_some();
    let source = match (stored, !envs.is_empty()) {
        (true, true) => format!("a stored key and {}", envs.join(", ")),
        (true, false) => "a stored key in ~/.grok/auth.json".to_string(),
        (false, true) => format!("the environment variable {}", envs.join(", ")),
        (false, false) => "an unknown source".to_string(),
    };
    let mut out = format!(
        "{} is connected via {}.\n\nModels ({}):\n",
        spec.name,
        source,
        models.len()
    );
    for model in models.iter().take(30) {
        out.push_str(&format!("  /model {:<28} {}\n", model.slug, model.name));
    }
    if models.len() > 30 {
        out.push_str(&format!("  … and {} more (select with /model)\n", models.len() - 30));
    }
    out.push_str("\nThe model picker refreshes automatically after connect and disconnect.");
    out
}

fn unknown_provider(input: &str) -> String {
    let ids: Vec<&str> = providers::builtin_providers().iter().map(|s| s.id).collect();
    format!("Unknown provider '{input}'. Built-in: {}", ids.join(", "))
}

fn connect_instructions(spec: &ProviderSpec) -> String {
    format!(
        "{name} is not connected.\n\n\
         In-session connect opened the key dialog; on the welcome screen, add it from a \
         terminal (it will ask for the key — secrets never land in scrollback):\n\
         \x20 grok connect {id}\n\n\
         Headless / CI:\n\
         \x20 echo \"$KEY\" | grok connect {id}\n\
         \x20 grok connect {id} --api-key \"$KEY\"\n\n\
         Or set {env} and restart Grok.\n\n\
         Get a key: {docs}",
        name = spec.name,
        id = spec.id,
        env = spec.env_keys.join(" or "),
        docs = spec.docs_url,
    )
}