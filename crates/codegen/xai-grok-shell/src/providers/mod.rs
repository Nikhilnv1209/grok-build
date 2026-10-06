//! Built-in open-source inference providers (Umans, DeepSeek, OpenCode, Command Code).
//!
//! Connect stores an API key in `auth.json` under `provider:<id>`. The catalog
//! describes those models so `/model` and the sampler stay unchanged.
//!
//! Model lists come from the checked-in models.dev snapshot for Umans /
//! DeepSeek / OpenCode and from a checked-in open-weight list for Command
//! Code. Nothing here reaches the network: there is no refresh, so model
//! metadata is whatever ships in the binary.

mod catalog;

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::agent::config::{EnvKeys, ModelEntry};
use crate::auth::{AuthProviderRef, AuthStore};
use crate::sampling::ApiBackend;
use xai_grok_sampler::AuthScheme;
use xai_grok_sampling_types::ReasoningEffort;

pub use catalog::{
    base_url, builtin_providers, commandcode_models, snapshot_models, ProviderSpec,
};

const PROVIDER_SCOPE_PREFIX: &str = "provider:";

/// One catalog row hydrated after the provider is connected.
#[derive(Clone, Debug)]
pub struct ProviderModel {
    pub slug: String,
    pub model: String,
    pub name: String,
    pub description: String,
    pub context_window: u64,
    pub max_output_tokens: Option<u32>,
    pub supports_reasoning_effort: bool,
    pub default_reasoning: Option<ReasoningEffort>,
    pub reasoning_efforts: Vec<ReasoningEffort>,
    pub api_backend: ApiBackend,
    pub auth_scheme: AuthScheme,
}

impl ProviderSpec {
    pub fn matches(&self, input: &str) -> bool {
        let needle = input.trim();
        self.id.eq_ignore_ascii_case(needle)
            || self
                .aliases
                .iter()
                .any(|alias| alias.eq_ignore_ascii_case(needle))
    }

    pub fn scope_key(&self) -> String {
        provider_scope(self.id)
    }
}

pub fn provider_scope(id: &str) -> String {
    format!("{PROVIDER_SCOPE_PREFIX}{id}")
}

pub fn find_provider(input: &str) -> Option<&'static ProviderSpec> {
    let needle = input.trim();
    builtin_providers()
        .iter()
        .find(|spec| spec.matches(needle))
}

/// Friendly picker label for an auth scope: `provider:deepseek` → `"DeepSeek"`.
/// `None` for non-builtin scopes (custom `auth_provider`/`model_provider`
/// refs), letting callers fall back to their own "custom" marker.
pub fn provider_label_for_scope(scope: &str) -> Option<&'static str> {
    scope
        .strip_prefix(PROVIDER_SCOPE_PREFIX)
        .and_then(find_provider)
        .map(|spec| spec.name)
}

pub fn env_belongs_to_provider(env_name: &str) -> Option<&'static ProviderSpec> {
    builtin_providers()
        .iter()
        .find(|spec| spec.env_keys.iter().any(|key| *key == env_name))
}

/// All models a provider can serve today, from the checked-in catalog.
pub fn provider_models(spec: &ProviderSpec) -> Vec<ProviderModel> {
    match spec.id {
        "commandcode" => commandcode_models(),
        _ => snapshot_models(spec),
    }
}

/// `auth.json` lookup is skipped in `cargo test` unless `GROK_HOME` is set, so
/// a developer's real credentials cannot leak into catalog assertions.
fn stored_lookup_enabled() -> bool {
    !cfg!(test) || std::env::var_os("GROK_HOME").is_some()
}

pub fn read_stored_key(provider_id: &str) -> Option<String> {
    if !stored_lookup_enabled() {
        return None;
    }
    let path = xai_grok_config::grok_home().join("auth.json");
    let store = crate::auth::read_auth_json(&path).ok()?;
    store
        .get(&provider_scope(provider_id))
        .map(|auth| auth.key.clone())
        .filter(|key| !key.trim().is_empty())
}

pub fn stored_key_for_env_keys(env_key: &EnvKeys) -> Option<String> {
    for name in env_key.names() {
        if let Some(spec) = env_belongs_to_provider(name)
            && let Some(key) = read_stored_key(spec.id)
        {
            return Some(key);
        }
    }
    None
}

pub fn is_connected(spec: &ProviderSpec) -> bool {
    env_is_set(spec) || read_stored_key(spec.id).is_some()
}

/// Environment variables of the provider that are currently set (non-blank),
/// in declared priority order. Lets the UI say *why* a provider is connected.
pub fn set_env_names(spec: &ProviderSpec) -> Vec<&'static str> {
    spec.env_keys
        .iter()
        .copied()
        .filter(|name| std::env::var(name).is_ok_and(|v| !v.trim().is_empty()))
        .collect()
}

fn env_is_set(spec: &ProviderSpec) -> bool {
    !set_env_names(spec).is_empty()
}

pub fn store_key(provider_id: &str, api_key: &str) -> std::io::Result<()> {
    let spec = find_provider(provider_id).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("unknown provider '{provider_id}'"),
        )
    })?;
    let path = xai_grok_config::grok_home().join("auth.json");
    let mut map = crate::auth::read_auth_json_or_empty_recovering_corrupt(&path)?;
    map.insert(
        spec.scope_key(),
        crate::auth::GrokAuth {
            key: api_key.trim().to_owned(),
            auth_mode: crate::auth::AuthMode::ApiKey,
            ..Default::default()
        },
    );
    crate::auth::write_auth_json(&path, &map)
}

pub fn clear_key(provider_id: &str) -> std::io::Result<bool> {
    let spec = find_provider(provider_id).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("unknown provider '{provider_id}'"),
        )
    })?;
    let path = xai_grok_config::grok_home().join("auth.json");
    let Ok(mut map) = crate::auth::read_auth_json(&path) else {
        return Ok(false);
    };
    let removed = map.remove(&spec.scope_key()).is_some();
    if !removed {
        return Ok(false);
    }
    if map.is_empty() {
        let _ = std::fs::remove_file(&path);
    } else {
        crate::auth::write_auth_json(&path, &map)?;
    }
    Ok(true)
}

pub fn stored_credentials_fingerprint(store: &AuthStore) -> u64 {
    let mut hasher = DefaultHasher::new();
    for (scope, auth) in store {
        if scope.starts_with(PROVIDER_SCOPE_PREFIX) {
            scope.hash(&mut hasher);
            auth.key.hash(&mut hasher);
        }
    }
    hasher.finish()
}

/// Whether a model speaks the DeepSeek OpenAI-compatible dialect: a
/// `thinking: { type: "enabled" }` envelope alongside `reasoning_effort`, plus
/// `reasoning_content` on every replayed assistant message. The decision lives
/// here so no core crate ever sniffs model ids or provider names.
fn is_deepseek_class(model: &ModelEntry) -> bool {
    if model.info().model.to_ascii_lowercase().contains("deepseek") {
        return true;
    }
    if let Some(ref provider) = model.auth_provider
        && provider.name == provider_scope("deepseek")
    {
        return true;
    }
    if let Some(ref env) = model.env_key {
        return env.names().iter().any(|name| {
            env_belongs_to_provider(name).is_some_and(|spec| spec.id == "deepseek")
        });
    }
    false
}

/// Provider-class request shaping for the chat-completions backend:
/// `Some(thinking)` marks a DeepSeek-style reasoning model so the mapper emits
/// the `thinking` envelope and normalizes assistant `reasoning_content`.
/// `None` keeps the plain OpenAI behavior.
pub fn model_thinking(
    model: &ModelEntry,
) -> Option<xai_grok_sampling_types::ChatThinking> {
    use xai_grok_sampling_types::{ChatThinking, ChatThinkingType};
    let info = model.info();
    if info.api_backend != ApiBackend::ChatCompletions
        || info.reasoning_effort.is_none()
        || !is_deepseek_class(model)
    {
        return None;
    }
    Some(ChatThinking {
        r#type: ChatThinkingType::Enabled,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_provider_accepts_aliases() {
        assert_eq!(find_provider("umans").map(|s| s.id), Some("umans"));
        assert_eq!(find_provider("umans-ai").map(|s| s.id), Some("umans"));
        assert_eq!(
            find_provider("umans-ai-coding-plan").map(|s| s.id),
            Some("umans")
        );
        assert_eq!(find_provider("DeepSeek").map(|s| s.id), Some("deepseek"));
        assert_eq!(find_provider("opencode").map(|s| s.id), Some("opencode"));
        assert_eq!(
            find_provider("command-code").map(|s| s.id),
            Some("commandcode")
        );
        assert_eq!(find_provider("cmdc").map(|s| s.id), Some("commandcode"));
        assert!(find_provider("openai").is_none());
    }

    #[test]
    fn env_maps_to_provider() {
        assert_eq!(
            env_belongs_to_provider("DEEPSEEK_API_KEY").map(|s| s.id),
            Some("deepseek")
        );
        assert_eq!(
            env_belongs_to_provider("UMANS_AI_CODING_PLAN_API_KEY").map(|s| s.id),
            Some("umans")
        );
        assert_eq!(
            env_belongs_to_provider("UMANS_AI_API_KEY").map(|s| s.id),
            Some("umans")
        );
        assert_eq!(
            env_belongs_to_provider("CMD_API_KEY").map(|s| s.id),
            Some("commandcode")
        );
    }

    #[test]
    fn provider_models_commandcode_uses_checked_in_list() {
        let cmd = find_provider("commandcode").unwrap();
        let models = provider_models(cmd);
        assert!(!models.is_empty());
        assert!(models.iter().any(|m| m.model.contains("Kimi")));
    }

    /// DeepSeek-family models get the `thinking` envelope; other open-source
    /// models (e.g. GLM via OpenCode Zen) must not. Detection reads the
    /// model's own metadata, so it holds for any entry the user configures.
    #[serial_test::serial]
    #[test]
    fn model_thinking_applies_only_to_deepseek_class() {
        use xai_grok_sampling_types::{ChatThinking, ChatThinkingType};
        const RELEVANT: &[&str] = &["DEEPSEEK_API_KEY", "OPENCODE_API_KEY"];
        let saved: Vec<(String, Option<String>)> = RELEVANT
            .iter()
            .map(|key| ((*key).to_string(), std::env::var(key).ok()))
            .collect();
        for key in RELEVANT {
            // SAFETY: serial_test::serial keeps provider env out of view.
            unsafe { std::env::remove_var(key) };
        }

        unsafe { std::env::set_var("DEEPSEEK_API_KEY", "sk-test") };
        unsafe { std::env::set_var("OPENCODE_API_KEY", "sk-test") };

        let deepseek = reasoning_entry("deepseek-v4-flash", "deepseek", &["DEEPSEEK_API_KEY"]);
        assert_eq!(
            model_thinking(&deepseek),
            Some(ChatThinking {
                r#type: ChatThinkingType::Enabled
            })
        );
        let glm = reasoning_entry("glm-5.2", "opencode", &["OPENCODE_API_KEY"]);
        assert_eq!(model_thinking(&glm), None, "GLM is not DeepSeek-class");

        for (key, value) in saved {
            // SAFETY: see above; env is restored for other tests.
            match value {
                Some(value) => unsafe { std::env::set_var(key, value) },
                None => unsafe { std::env::remove_var(key) },
            }
        }
    }

    /// A minimal `ModelEntry` carrying just what `model_thinking` inspects:
    /// the wire model id, the chat-completions backend, a reasoning effort,
    /// and the provider env key.
    fn reasoning_entry(model: &str, provider: &str, env_keys: &[&str]) -> ModelEntry {
        ModelEntry {
            model: model.to_string(),
            api_backend: ApiBackend::ChatCompletions,
            reasoning_effort: Some(ReasoningEffort::High),
            auth_provider: Some(AuthProviderRef::fail_closed(provider_scope(provider))),
            env_key: Some(EnvKeys::new(env_keys.iter().copied())),
            ..Default::default()
        }
    }
}