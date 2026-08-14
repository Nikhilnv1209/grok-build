//! `grok connect` / `grok disconnect` — add or remove open-source providers.

use std::io::{self, IsTerminal, Read, Write};

use anyhow::{bail, Result};
use clap::Args;
use xai_grok_shell::providers::{self, ProviderSpec};

#[derive(Debug, Args, Clone)]
pub struct ConnectArgs {
    /// Provider id (`umans`, `deepseek`, `opencode`, `commandcode`). Omit to list providers.
    pub provider: Option<String>,
    /// API key to store. If omitted, reads a line from stdin.
    #[arg(long = "api-key", env = "GROK_CONNECT_API_KEY")]
    pub api_key: Option<String>,
    /// List built-in providers and whether they are connected.
    #[arg(long, short = 'l')]
    pub list: bool,
    /// Refresh a provider's model catalog from the network (Command Code).
    #[arg(long)]
    pub refresh: bool,
}

#[derive(Debug, Args, Clone)]
pub struct DisconnectArgs {
    /// Provider id to disconnect (`umans`, `deepseek`, `opencode`, `commandcode`).
    pub provider: String,
}

pub fn run_connect(args: ConnectArgs) -> Result<()> {
    if args.list || args.provider.is_none() {
        print_status();
        if args.provider.is_none() && !args.list {
            println!();
            println!("Connect a provider:");
            println!("  grok connect umans");
            println!("  grok connect deepseek --api-key \"$DEEPSEEK_API_KEY\"");
            println!("  grok connect opencode");
            println!("  grok connect commandcode");
            println!();
            println!("Or set the environment variable and start grok — no connect step needed.");
        }
        return Ok(());
    }

    let input = args.provider.as_deref().unwrap_or_default();
    let spec = require_provider(input)?;
    let key = match args.api_key {
        Some(key) => key,
        None if args.refresh => match providers::read_stored_key(spec.id) {
            Some(key) => key,
            None => prompt_api_key(spec)?,
        },
        None => prompt_api_key(spec)?,
    };
    let key = key.trim();
    if key.is_empty() {
        bail!("API key is empty");
    }

    // Command Code exposes its own `/v1/models`; fetch the real list whenever
    // we have a key (connect with a key, or `--refresh` to update an existing
    // one). Failing to fetch is non-fatal — the snapshot fallback still
    // hydrates — but the live catalog is preferred once reachable.
    if spec.id == "commandcode" || args.refresh {
        match xai_grok_shell::providers::refresh_provider_models(spec, key) {
            Ok(()) => println!("Refreshed {} model list from the provider.", spec.name),
            Err(error) => {
                println!("Could not refresh model list ({error}); using the bundled catalog.");
            }
        }
    }

    providers::store_key(spec.id, key)?;
    let env_list = spec.env_keys.join(", ");
    println!("Connected {}.", spec.name);
    println!("  stored in ~/.grok/auth.json as provider:{}", spec.id);
    println!("  env fallback: {}", env_list);
    println!("  docs: {}", spec.docs_url);
    println!();
    let models = providers::provider_models(spec);
    println!("Models ({}):", models.len());
    for model in models.iter().take(40) {
        println!("  /model {}    {}", model.slug, model.name);
    }
    if models.len() > 40 {
        println!("  … and {} more (select with /model)", models.len() - 40);
    }
    Ok(())
}

pub fn run_disconnect(args: DisconnectArgs) -> Result<()> {
    let spec = require_provider(&args.provider)?;
    if providers::clear_key(spec.id)? {
        println!("Disconnected {}.", spec.name);
    } else {
        println!(
            "No stored API key for {}. Environment variables {} are unchanged.",
            spec.name,
            spec.env_keys.join(", ")
        );
    }
    Ok(())
}

fn require_provider(input: &str) -> Result<&'static ProviderSpec> {
    providers::find_provider(input).ok_or_else(|| {
        let ids: Vec<&str> = providers::builtin_providers().iter().map(|s| s.id).collect();
        anyhow::anyhow!("Unknown provider '{input}'. Built-in: {}", ids.join(", "))
    })
}

fn print_status() {
    println!("Open-source providers");
    println!();
    for spec in providers::builtin_providers() {
        let via_env = env_is_set(spec);
        let via_store = providers::read_stored_key(spec.id).is_some();
        let status = match (via_store, via_env) {
            (true, true) => "connected (auth.json + env)",
            (true, false) => "connected (auth.json)",
            (false, true) => "connected (env)",
            (false, false) => "not connected",
        };
        let model_count = providers::provider_models(spec).len();
        println!(
            "  {:<12}  {}  — {}  ({} models)",
            spec.id, spec.name, status, model_count
        );
        println!("               env {}  {}", spec.env_keys.join(", "), spec.docs_url);
    }
}

fn env_is_set(spec: &ProviderSpec) -> bool {
    spec.env_keys
        .iter()
        .any(|name| std::env::var(name).is_ok_and(|v| !v.trim().is_empty()))
}

fn prompt_api_key(spec: &ProviderSpec) -> Result<String> {
    let env_note = if providers::is_connected(spec) {
        format!(
            "\nNote: {} is already set in the environment or authored in ~/.grok/auth.json; a new key below still gets stored and wins.",
            spec.env_keys.join(" / ")
        )
    } else {
        String::new()
    };
    eprint!(
        "Enter {} API key ({}):{} ",
        spec.name,
        spec.env_keys.join(" or "),
        env_note
    );
    io::stderr().flush()?;

    if !io::stdin().is_terminal() {
        // Piped input (CI / `echo $KEY | grok connect …`): read one visible line.
        let mut line = String::new();
        io::stdin().read_line(&mut line)?;
        return Ok(line);
    }

    // Interactive: disable terminal echo so the secret never appears on
    // screen or in scrollback. Read raw bytes until Enter.
    use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
    enable_raw_mode()?;
    let mut secret = String::new();
    let read_result = (|| -> std::io::Result<()> {
        let mut byte = [0u8; 1];
        loop {
            let read = io::stdin().read(&mut byte)?;
            if read == 0 {
                return Ok(());
            }
            match byte[0] {
                b'\r' | b'\n' => return Ok(()),
                0x7f | 0x08 => {
                    secret.pop();
                }
                b => secret.push(b as char),
            }
        }
    })();
    let _ = disable_raw_mode();
    eprintln!();
    read_result?;
    Ok(secret)
}