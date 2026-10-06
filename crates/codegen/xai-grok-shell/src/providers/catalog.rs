//! Built-in provider registry.
//!
//! Umans / DeepSeek / OpenCode model lists come from the checked-in snapshot of
//! https://models.dev/api.json (`data/models.dev.subset.json`), so context
//! windows, reasoning flags, and temperature rules stay current without a
//! second hand-written table. The snapshot ships with the binary; nothing here
//! is fetched at runtime. Command Code is not on models.dev, so it uses a
//! checked-in open-weight list.

use std::sync::OnceLock;

use indexmap::IndexMap;
use serde::Deserialize;

use crate::sampling::ApiBackend;
use xai_grok_sampler::AuthScheme;
use xai_grok_sampling_types::ReasoningEffort;

use super::ProviderModel;

const MODELS_DEV_SNAPSHOT: &str = include_str!("data/models.dev.subset.json");

// ── Provider specs ─────────────────────────────────────────────────────────

/// Static per-provider metadata. Model lists are resolved at hydrate time by
/// `super::provider_models`, so the registry itself never goes stale.
#[derive(Clone, Debug)]
pub struct ProviderSpec {
    pub id: &'static str,
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    /// Primary env var, then alternates (all accepted, first set wins).
    pub env_keys: &'static [&'static str],
    pub docs_url: &'static str,
    /// Base URL used when no snapshot entry supplies one (models.dev `api`
    /// field wins when present).
    pub default_base_url: &'static str,
    /// Which check-in snapshot keys back this provider. Empty = provider
    /// maintains its own catalog (Command Code).
    pub models_dev_ids: &'static [&'static str],
    pub default_backend: ApiBackend,
    pub auth_scheme: AuthScheme,
}

pub fn builtin_providers() -> &'static [ProviderSpec] {
    &[
        UMANS,
        DEEPSEEK,
        OPENCODE,
        COMMAND_CODE,
    ]
}

const UMANS: ProviderSpec = ProviderSpec {
    id: "umans",
    name: "Umans AI",
    aliases: &["umans-ai", "umans-ai-coding-plan"],
    env_keys: &["UMANS_AI_CODING_PLAN_API_KEY", "UMANS_AI_API_KEY"],
    docs_url: "https://app.umans.ai/offers/code/docs",
    default_base_url: "https://api.code.umans.ai/v1",
    models_dev_ids: &["umans-ai-coding-plan", "umans-ai"],
    default_backend: ApiBackend::ChatCompletions,
    auth_scheme: AuthScheme::Bearer,
};

const DEEPSEEK: ProviderSpec = ProviderSpec {
    id: "deepseek",
    name: "DeepSeek",
    aliases: &[],
    env_keys: &["DEEPSEEK_API_KEY"],
    docs_url: "https://api-docs.deepseek.com",
    default_base_url: "https://api.deepseek.com",
    models_dev_ids: &["deepseek"],
    default_backend: ApiBackend::ChatCompletions,
    auth_scheme: AuthScheme::Bearer,
};

const OPENCODE: ProviderSpec = ProviderSpec {
    id: "opencode",
    name: "OpenCode Zen",
    aliases: &["opencode-zen", "zen"],
    env_keys: &["OPENCODE_API_KEY"],
    docs_url: "https://opencode.ai/docs/zen",
    default_base_url: "https://opencode.ai/zen/v1",
    models_dev_ids: &["opencode"],
    default_backend: ApiBackend::ChatCompletions,
    auth_scheme: AuthScheme::Bearer,
};

const COMMAND_CODE: ProviderSpec = ProviderSpec {
    id: "commandcode",
    name: "Command Code",
    aliases: &["command-code", "cmdc"],
    env_keys: &["CMD_API_KEY"],
    docs_url: "https://commandcode.ai/docs/provider",
    default_base_url: "https://api.commandcode.ai/provider/v1",
    models_dev_ids: &[],
    default_backend: ApiBackend::ChatCompletions,
    auth_scheme: AuthScheme::Bearer,
};

// ── models.dev snapshot ────────────────────────────────────────────────────

/// Raw snapshot entries we care about.
#[derive(Clone, Deserialize)]
struct ModelsDevSnapshot {
    #[serde(flatten)]
    providers: IndexMap<String, ModelsDevProvider>,
}

#[derive(Clone, Deserialize)]
struct ModelsDevProvider {
    #[serde(rename = "api")]
    api: Option<String>,
    #[serde(rename = "models")]
    models: IndexMap<String, ModelsDevModel>,
}

#[derive(Clone, Deserialize)]
struct ModelsDevModel {
    #[serde(rename = "id")]
    id: String,
    #[serde(rename = "name")]
    name: String,
    #[serde(rename = "description")]
    description: Option<String>,
    #[serde(rename = "reasoning")]
    reasoning: Option<bool>,
    #[serde(rename = "reasoning_options")]
    reasoning_options: Option<Vec<ReasoningOption>>,
    #[serde(rename = "temperature")]
    temperature: Option<bool>,
    #[serde(rename = "limit")]
    limit: Option<ModelsDevLimit>,
    #[serde(rename = "status")]
    status: Option<String>,
}

/// One models.dev `reasoning_options` entry: `{"type":"toggle"}` (reasoning
/// on/off → low/high) or `{"type":"effort","values":["low","max",…]}`.
#[derive(Clone, Deserialize)]
struct ReasoningOption {
    #[serde(rename = "type")]
    r#type: String,
    #[serde(default, rename = "values")]
    values: Vec<String>,
}

#[derive(Clone, Deserialize)]
struct ModelsDevLimit {
    #[serde(rename = "context")]
    context: Option<u64>,
    #[serde(rename = "output")]
    output: Option<u64>,
}

fn parse_snapshot(json: &str) -> Option<ModelsDevSnapshot> {
    serde_json::from_str(json)
        .map_err(|error| tracing::warn!(%error, "models.dev catalog unparseable"))
        .ok()
}

fn checked_in_snapshot() -> &'static ModelsDevSnapshot {
    static SNAPSHOT: OnceLock<ModelsDevSnapshot> = OnceLock::new();
    SNAPSHOT.get_or_init(|| {
        serde_json::from_str(MODELS_DEV_SNAPSHOT)
            .expect("models.dev.subset.json must parse (checked in, dev error)")
    })
}

/// The catalog in force: the checked-in snapshot only. There is no live
/// fetch -- provider model metadata is compiled in, so model resolution
/// never touches the network.
fn active_snapshot() -> std::sync::Arc<ModelsDevSnapshot> {
    static FALLBACK: OnceLock<std::sync::Arc<ModelsDevSnapshot>> = OnceLock::new();
    FALLBACK
        .get_or_init(|| std::sync::Arc::new(checked_in_snapshot().clone()))
        .clone()
}

/// Resolve a provider's base URL: the models.dev `api` field wins over the
/// spec default so we never drift from the registry.
pub fn base_url(spec: &ProviderSpec) -> String {
    base_url_in(&active_snapshot(), spec)
}

fn base_url_in(snapshot: &ModelsDevSnapshot, spec: &ProviderSpec) -> String {
    for id in spec.models_dev_ids {
        if let Some(provider) = snapshot.providers.get(*id) {
            return provider
                .api
                .clone()
                .unwrap_or_else(|| spec.default_base_url.to_string());
        }
    }
    spec.default_base_url.to_string()
}

/// All catalog models for a provider, live-cache first, checked-in snapshot
/// as fallback. Merged across the provider's models.dev ids (Umans has both a
/// plan and an org entry). Union by wire id.
pub fn snapshot_models(spec: &ProviderSpec) -> Vec<ProviderModel> {
    snapshot_models_in(&active_snapshot(), spec)
}

fn snapshot_models_in(snapshot: &ModelsDevSnapshot, spec: &ProviderSpec) -> Vec<ProviderModel> {
    let mut out: Vec<ProviderModel> = Vec::new();
    for id in spec.models_dev_ids {
        let Some(provider) = snapshot.providers.get(*id) else {
            tracing::warn!(provider = ?id, "models.dev snapshot missing provider entry");
            continue;
        };
        for (key, model) in &provider.models {
            if model.status.as_deref() == Some("deprecated") {
                continue;
            }
            let model_id = model.id.clone();
            if out.iter().any(|m| m.model == model_id) {
                continue;
            }
            let slug = slug_for(spec.id, &key);
            let (supports_reasoning, reasoning_efforts) = reasoning_map(model);
            out.push(ProviderModel {
                slug,
                model: model_id,
                name: model.name.clone(),
                description: model.description.clone().unwrap_or_default(),
                context_window: model.limit.as_ref().and_then(|l| l.context).unwrap_or(200_000),
                max_output_tokens: model
                    .limit
                    .as_ref()
                    .and_then(|l| l.output.and_then(|v| u32::try_from(v).ok())),
                supports_reasoning_effort: supports_reasoning,
                default_reasoning: reasoning_efforts.first().copied(),
                reasoning_efforts,
                api_backend: spec.default_backend.clone(),
                auth_scheme: spec.auth_scheme,
            });
        }
    }
    out
}

fn slug_for(provider_id: &str, key: &str) -> String {
    let slug = key.replace('/', "-");
    if slug.starts_with(provider_id) {
        slug
    } else {
        format!("{provider_id}-{slug}")
    }
}

fn reasoning_map(model: &ModelsDevModel) -> (bool, Vec<ReasoningEffort>) {
    let mut efforts: Vec<ReasoningEffort> = Vec::new();
    if let Some(options) = &model.reasoning_options {
        for option in options {
            match option.r#type.as_str() {
                "toggle" => {
                    efforts.push(ReasoningEffort::Low);
                    efforts.push(ReasoningEffort::High);
                }
                "effort" => {
                    for v in &option.values {
                        if let Ok(effort) = v.parse::<ReasoningEffort>() {
                            efforts.push(effort);
                        }
                    }
                }
                other => {
                    tracing::warn!(r#type = other, "unknown models.dev reasoning option type");
                }
            }
        }
    }
    if model.reasoning.unwrap_or(false) && efforts.is_empty() {
        // `[]` reasoning_options means "reasons, but there is no dial".
        efforts.push(ReasoningEffort::High);
    }
    let mut seen: Vec<ReasoningEffort> = Vec::new();
    for effort in efforts {
        if !seen.contains(&effort) {
            seen.push(effort);
        }
    }
    (model.reasoning.unwrap_or(false), seen)
}

fn effort_as_str(e: &ReasoningEffort) -> &'static str {
    match e {
        ReasoningEffort::None => "none",
        ReasoningEffort::Minimal => "minimal",
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::Xhigh => "xhigh",
        ReasoningEffort::Max => "max",
    }
}

// ── Command Code catalog ───────────────────────────────────────────────────

/// The open-weight models Command Code serves over the OpenAI-compatible
/// route. Checked in; nothing replaces it at runtime.
const COMMAND_CODE_FALLBACK: &[(&str, &str, u64)] = &[
    ("deepseek/deepseek-v4-flash", "DeepSeek V4 Flash", 1_000_000),
    ("deepseek/deepseek-v4-pro", "DeepSeek V4 Pro", 1_000_000),
    ("moonshotai/Kimi-K3", "Kimi K3", 1_000_000),
    ("moonshotai/Kimi-K2.7-Code", "Kimi K2.7 Code", 256_000),
    ("zai-org/GLM-5.2", "GLM-5.2", 1_000_000),
    ("MiniMaxAI/MiniMax-M3", "MiniMax M3", 1_000_000),
    ("Qwen/Qwen3.8-Max", "Qwen 3.8 Max", 1_000_000),
    ("xai/grok-4.6", "Grok 4.6", 500_000),
];

fn commandcode_backend(model_id: &str) -> ApiBackend {
    // Claude models on Command Code 400 on the chat-completions route.
    if model_id.starts_with("claude-")
        || model_id.starts_with("anthropic/")
    {
        ApiBackend::Messages
    } else {
        ApiBackend::ChatCompletions
    }
}

/// Checked-in open-weight catalog for Command Code. Never fetched.
pub fn commandcode_models() -> Vec<ProviderModel> {
    COMMAND_CODE_FALLBACK
        .iter()
        .map(|(id, name, ctx)| ProviderModel {
            slug: slug_for("commandcode", id),
            model: (*id).to_string(),
            name: (*name).to_string(),
            description: "Command Code (fallback catalog)".to_string(),
            context_window: *ctx,
            max_output_tokens: None,
            supports_reasoning_effort: true,
            default_reasoning: Some(ReasoningEffort::High),
            reasoning_efforts: vec![
                ReasoningEffort::Low,
                ReasoningEffort::Medium,
                ReasoningEffort::High,
            ],
            api_backend: commandcode_backend(id),
            auth_scheme: AuthScheme::Bearer,
        })
        .collect()
}

#[cfg(test)]
impl ModelsDevSnapshot {
    fn umans_plan_model_count(&self) -> usize {
        self.providers
            .get("umans-ai-coding-plan")
            .map(|p| p.models.len())
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_parses_with_providers() {
        let snap = checked_in_snapshot();
        assert!(snap.umans_plan_model_count() >= 5, "umans-ai-coding-plan models");
        assert!(snap.providers.contains_key("deepseek"));
        assert!(snap.providers.contains_key("opencode"));
    }

    /// Snapshot parsing drives every downstream metadata field (slugs, base
    /// URLs, reasoning menus), so a new provider entry needs no code change.
    #[test]
    fn parsed_snapshot_drives_slugs_and_base_url() {
        let spec = builtin_providers()
            .iter()
            .find(|s| s.id == "deepseek")
            .unwrap();
        let live = r#"{"deepseek":{"api":"https://example.invalid/v1","models":{
            "brand-new-model":{"id":"brand-new-model","name":"Brand New","reasoning":true}
        }}}"#;
        let parsed = parse_snapshot(live).expect("test fixture parses");
        let models = snapshot_models_in(&parsed, spec);
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].slug, "deepseek-brand-new-model");
        assert_eq!(
            base_url_in(&parsed, spec),
            "https://example.invalid/v1",
            "the `api` field wins over the spec default"
        );
    }

    #[test]
    fn snapshot_models_depth_first() {
        let deepseek = builtin_providers()
            .iter()
            .find(|s| s.id == "deepseek")
            .unwrap();
        let models = snapshot_models(deepseek);
        assert!(
            models.iter().any(|m| m.slug == "deepseek-v4-flash"),
            "deepseek-v4-flash must hydrate"
        );
        assert!(
            models.iter().any(|m| m.slug == "deepseek-v4-pro"),
            "deepseek-v4-pro must hydrate"
        );
        assert!(
            models.iter().all(|m| m.context_window == 1_000_000),
            "snapshot context window should ride through"
        );
    }

    #[test]
    fn opencode_slugs_are_namespaced() {
        let opencode = builtin_providers()
            .iter()
            .find(|s| s.id == "opencode")
            .unwrap();
        let models = snapshot_models(opencode);
        assert!(models.len() >= 10, "got {}", models.len());
        assert!(
            models.iter().any(|m| m.slug == "opencode-glm-5.2"),
            "glm-5.2 should become opencode-glm-5.2"
        );
        assert!(
            models.iter().all(|m| m.slug.starts_with("opencode-")),
            "all opencode slugs namespaced"
        );
    }

    #[test]
    fn umans_merges_plan_and_org_entries() {
        let umans = builtin_providers().iter().find(|s| s.id == "umans").unwrap();
        let models = snapshot_models(umans);
        let wire_ids: std::collections::HashSet<&str> =
            models.iter().map(|m| m.model.as_str()).collect();
        assert!(
            wire_ids.contains("umans-coder") && wire_ids.contains("umans-kimi-k3"),
            "union must include both plan and org entries: {:?}",
            wire_ids
        );
        // Union by wire id: no duplicate slugs.
        let mut slugs: Vec<&str> = models.iter().map(|m| m.slug.as_str()).collect();
        let len = slugs.len();
        slugs.sort();
        slugs.dedup();
        assert_eq!(slugs.len(), len, "no duplicate slugs");
    }

    #[test]
    fn commandcode_catalog_is_open_weight_chat_models() {
        let models = commandcode_models();
        assert!(!models.is_empty());
        assert!(
            models.iter().all(|m| m.api_backend == ApiBackend::ChatCompletions),
            "checked-in catalog only carries OpenAI-compatible open models"
        );
        assert!(models.iter().any(|m| m.model.contains("Kimi")));
    }

    #[test]
    fn reasoning_effort_roundtrip_strings() {
        for e in [
            ReasoningEffort::None,
            ReasoningEffort::Minimal,
            ReasoningEffort::Low,
            ReasoningEffort::Medium,
            ReasoningEffort::High,
            ReasoningEffort::Xhigh,
            ReasoningEffort::Max,
        ] {
            assert_eq!(e.as_str(), effort_as_str(&e));
        }
    }

    /// Regression: models.dev effort lists (`{"type":"effort","values":[...]}`)
    /// were silently parsed as toggles, collapsing every DeepSeek menu to
    /// low/high and dropping `max`.
    #[test]
    fn effort_values_survive_deserialization() {
        let deepseek = builtin_providers().iter().find(|s| s.id == "deepseek").unwrap();
        let flash = snapshot_models(deepseek)
            .into_iter()
            .find(|m| m.model == "deepseek-v4-flash")
            .expect("deepseek-v4-flash must load from the snapshot");
        assert_eq!(
            flash.reasoning_efforts,
            vec![ReasoningEffort::Low, ReasoningEffort::High, ReasoningEffort::Max],
            "models.dev declares low/high/max for deepseek-v4-flash"
        );

        let opencode = builtin_providers().iter().find(|s| s.id == "opencode").unwrap();
        let free = snapshot_models(opencode)
            .into_iter()
            .find(|m| m.model == "deepseek-v4-flash-free")
            .expect("opencode deepseek-v4-flash-free must load from the snapshot");
        assert_eq!(
            free.reasoning_efforts,
            vec![ReasoningEffort::Low, ReasoningEffort::High, ReasoningEffort::Max],
            "opencode deepseek-v4-flash-free must keep its max-effort option"
        );
        let pro = snapshot_models(opencode)
            .into_iter()
            .find(|m| m.model == "deepseek-v4-pro")
            .expect("opencode deepseek-v4-pro must load from the snapshot");
        assert_eq!(
            pro.reasoning_efforts,
            vec![ReasoningEffort::Low, ReasoningEffort::High, ReasoningEffort::Max],
            "toggle + effort merge must dedup to low/high/max"
        );
    }
}