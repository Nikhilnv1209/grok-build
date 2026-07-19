//! Static command-prefix arity table (ported from opencode's BashArity).
//!
//! No AI — pure lookup. Identifies the "human-understandable command" from
//! shell tokens so allow-patterns scope to the right granularity: not too
//! narrow (re-prompting for every arg variation) and not too broad (matching
//! unrelated commands that share a prefix).
//!
//! See `~/Documents/opencode/packages/opencode/src/permission/arity.ts`.

/// Return the command-unit prefix of `tokens` using the arity table.
///
/// Tries the longest token-prefix that is a table key; returns that many
/// tokens as the key's arity indicates (arity may be longer than the matched
/// key — e.g. `npm run` is arity 3, so `["npm","run","dev"]` returns all 3).
/// Unknown commands default to the first token. Empty input returns empty.
///
/// # Examples
/// ```
/// # use xai_grok_workspace::permission::arity::command_prefix;
/// assert_eq!(command_prefix(&["git", "checkout", "main"]), &["git", "checkout"][..]);
/// assert_eq!(command_prefix(&["npm", "run", "dev"]), &["npm", "run", "dev"][..]);
/// assert_eq!(command_prefix(&["aws", "s3", "ls"]), &["aws", "s3", "ls"][..]);
/// assert_eq!(command_prefix(&["unknown", "cmd"]), &["unknown"][..]);
/// ```
pub fn command_prefix<'a>(tokens: &'a [&'a str]) -> &'a [&'a str] {
    if tokens.is_empty() {
        return &[];
    }
    // Longest token-prefix that is a table key wins.
    for len in (1..=tokens.len()).rev() {
        let joined = tokens[..len].join(" ");
        if let Some(arity) = lookup(&joined) {
            let n = arity as usize;
            return &tokens[..n.min(tokens.len())];
        }
    }
    // Unknown — default to the first token (conservative: may re-prompt for
    // similar commands, but never matches too broadly).
    &tokens[..1]
}

/// Owned-String convenience wrapper for callers that own their tokens.
pub fn command_prefix_owned(tokens: &[String]) -> Vec<String> {
    let refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
    command_prefix(&refs)
        .iter()
        .map(|s: &&str| s.to_string())
        .collect()
}

/// Lookup a joined prefix string in the arity table.
fn lookup(joined: &str) -> Option<u8> {
    // Linear scan over a static slice — ~160 entries, called once per
    // permission prompt (not hot path). A HashMap/phf would be faster but
    // adds a dependency for negligible gain here.
    ARITY_TABLE
        .iter()
        .find(|(k, _)| *k == joined)
        .map(|(_, v)| *v)
}

/// The arity table: command-prefix → number of tokens that define the
/// command unit. Flags never count. Ported verbatim from opencode.
///
/// Only include a longer prefix if its arity differs from what the shorter
/// prefix already implies (opencode rule 4).
static ARITY_TABLE: &[(&str, u8)] = &[
    // arity 1 — single-token commands
    ("cat", 1),
    ("cd", 1),
    ("chmod", 1),
    ("chown", 1),
    ("cp", 1),
    ("echo", 1),
    ("env", 1),
    ("export", 1),
    ("grep", 1),
    ("kill", 1),
    ("killall", 1),
    ("ln", 1),
    ("ls", 1),
    ("mkdir", 1),
    ("mv", 1),
    ("ps", 1),
    ("pwd", 1),
    ("rm", 1),
    ("rmdir", 1),
    ("sleep", 1),
    ("source", 1),
    ("tail", 1),
    ("touch", 1),
    ("unset", 1),
    ("which", 1),
    // arity 2+ — multi-token command units
    ("aws", 3),
    ("az", 3),
    ("bazel", 2),
    ("brew", 2),
    ("bun", 2),
    ("bun run", 3),
    ("bun x", 3),
    ("cargo", 2),
    ("cargo add", 3),
    ("cargo run", 3),
    ("cdk", 2),
    ("cf", 2),
    ("cmake", 2),
    ("composer", 2),
    ("consul", 2),
    ("consul kv", 3),
    ("crictl", 2),
    ("deno", 2),
    ("deno task", 3),
    ("doctl", 3),
    ("docker", 2),
    ("docker builder", 3),
    ("docker compose", 3),
    ("docker container", 3),
    ("docker image", 3),
    ("docker network", 3),
    ("docker volume", 3),
    ("eksctl", 2),
    ("eksctl create", 3),
    ("firebase", 2),
    ("flyctl", 2),
    ("gcloud", 3),
    ("gh", 3),
    ("git", 2),
    ("git config", 3),
    ("git remote", 3),
    ("git stash", 3),
    ("go", 2),
    ("gradle", 2),
    ("helm", 2),
    ("heroku", 2),
    ("hugo", 2),
    ("ip", 2),
    ("ip addr", 3),
    ("ip link", 3),
    ("ip netns", 3),
    ("ip route", 3),
    ("kind", 2),
    ("kind create", 3),
    ("kubectl", 2),
    ("kubectl kustomize", 3),
    ("kubectl rollout", 3),
    ("kustomize", 2),
    ("make", 2),
    ("mc", 2),
    ("mc admin", 3),
    ("minikube", 2),
    ("mongosh", 2),
    ("mysql", 2),
    ("mvn", 2),
    ("ng", 2),
    ("npm", 2),
    ("npm exec", 3),
    ("npm init", 3),
    ("npm run", 3),
    ("npm view", 3),
    ("nvm", 2),
    ("nx", 2),
    ("openssl", 2),
    ("openssl req", 3),
    ("openssl x509", 3),
    ("pip", 2),
    ("pipenv", 2),
    ("pnpm", 2),
    ("pnpm dlx", 3),
    ("pnpm exec", 3),
    ("pnpm run", 3),
    ("poetry", 2),
    ("podman", 2),
    ("podman container", 3),
    ("podman image", 3),
    ("psql", 2),
    ("pulumi", 2),
    ("pulumi stack", 3),
    ("pyenv", 2),
    ("python", 2),
    ("rake", 2),
    ("rbenv", 2),
    ("redis-cli", 2),
    ("rustup", 2),
    ("serverless", 2),
    ("sfdx", 3),
    ("skaffold", 2),
    ("sls", 2),
    ("sst", 2),
    ("swift", 2),
    ("systemctl", 2),
    ("terraform", 2),
    ("terraform workspace", 3),
    ("tmux", 2),
    ("turbo", 2),
    ("ufw", 2),
    ("vault", 2),
    ("vault auth", 3),
    ("vault kv", 3),
    ("vercel", 2),
    ("volta", 2),
    ("wp", 2),
    ("yarn", 2),
    ("yarn dlx", 3),
    ("yarn run", 3),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arity_1_unknown_defaults_to_first_token() {
        assert_eq!(
            command_prefix(&["unknown", "command", "subcommand"]),
            &["unknown"][..]
        );
        assert_eq!(command_prefix(&["touch", "foo.txt"]), &["touch"][..]);
    }

    #[test]
    fn arity_2_two_token_commands() {
        assert_eq!(
            command_prefix(&["git", "checkout", "main"]),
            &["git", "checkout"][..]
        );
        assert_eq!(
            command_prefix(&["docker", "run", "nginx"]),
            &["docker", "run"][..]
        );
    }

    #[test]
    fn arity_3_three_token_commands() {
        assert_eq!(
            command_prefix(&["aws", "s3", "ls", "my-bucket"]),
            &["aws", "s3", "ls"][..]
        );
        assert_eq!(
            command_prefix(&["npm", "run", "dev", "script"]),
            &["npm", "run", "dev"][..]
        );
    }

    #[test]
    fn longest_match_wins_nested_prefixes() {
        assert_eq!(
            command_prefix(&["docker", "compose", "up", "service"]),
            &["docker", "compose", "up"][..]
        );
        assert_eq!(
            command_prefix(&["consul", "kv", "get", "config"]),
            &["consul", "kv", "get"][..]
        );
    }

    #[test]
    fn exact_length_matches() {
        assert_eq!(
            command_prefix(&["git", "checkout"]),
            &["git", "checkout"][..]
        );
        assert_eq!(
            command_prefix(&["npm", "run", "dev"]),
            &["npm", "run", "dev"][..]
        );
    }

    #[test]
    fn edge_cases() {
        let empty: &[&str] = &[];
        let result: &[&str] = command_prefix(empty);
        assert!(result.is_empty());
        assert_eq!(command_prefix(&["single"]), &["single"][..]);
        assert_eq!(command_prefix(&["git"]), &["git"][..]);
    }

    #[test]
    fn cargo_subcommands() {
        assert_eq!(command_prefix(&["cargo", "build"]), &["cargo", "build"][..]);
        assert_eq!(
            command_prefix(&["cargo", "test", "--lib"]),
            &["cargo", "test"][..]
        );
        assert_eq!(
            command_prefix(&["cargo", "add", "tokio"]),
            &["cargo", "add", "tokio"][..]
        );
    }

    #[test]
    fn owned_wrapper_matches_ref_version() {
        let tokens: Vec<String> = ["git", "checkout", "main"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            command_prefix_owned(&tokens),
            vec!["git".to_string(), "checkout".to_string()]
        );
    }

    #[test]
    fn arity_key_longer_than_arity_value() {
        // "git stash" has arity 3, so ["git", "stash"] (2 tokens) returns
        // both (clamped to tokens.len()).
        assert_eq!(command_prefix(&["git", "stash"]), &["git", "stash"][..]);
        // ["git", "stash", "pop"] → arity 3 → all 3.
        assert_eq!(
            command_prefix(&["git", "stash", "pop"]),
            &["git", "stash", "pop"][..]
        );
    }
}
