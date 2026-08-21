pub(crate) mod lsp_runtime;

pub(crate) const TEST_MODEL: &str = "test-model";

/// Permission bits (`mode & 0o777`) of `path`, for owner-only assertions.
#[cfg(unix)]
pub(crate) fn unix_mode(path: &std::path::Path) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path).unwrap().permissions().mode() & 0o777
}

/// Set `path`'s permission bits, e.g. to simulate umask-default dirs.
#[cfg(unix)]
pub(crate) fn set_unix_mode(path: &std::path::Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
}

/// Keep this crate's unit-test binary from writing synthetic events into
/// the real unified log; pre-main so the redirect beats the lazily-opened
/// writer. Integration binaries under `tests/` isolate via `TestSandbox`
/// homes instead.
#[ctor::ctor]
fn redirect_unified_log_for_tests() {
    xai_grok_telemetry::unified_log::redirect_to_temp_for_tests();
}

/// Pre-main hermetic bootstrap, run before any test (and before the harness
/// spawns worker threads):
///
/// 1. jsonwebtoken 10 refuses to guess a CryptoProvider when more than one
///    backend feature is enabled in the unified dep graph — product paths
///    install it lazily, but whichever test runs first would otherwise panic
///    depending on alphabetical order. Install deterministically here.
/// 2. Unset the built-in provider API-key env vars. The dev machine may
///    legitimately export them (env-based connect), which leaks hydrated
///    provider rows / BYOK credentials into tests that assert "no external
///    credentials". Tests that need them set explicit values via `EnvGuard`.
#[ctor::ctor]
fn hermetic_test_bootstrap() {
    let _ = jsonwebtoken::crypto::rust_crypto::DEFAULT_PROVIDER.install_default();
    for key in [
        "UMANS_AI_CODING_PLAN_API_KEY",
        "UMANS_AI_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENCODE_API_KEY",
        "CMD_API_KEY",
    ] {
        // SAFETY: ctor runs pre-main on a single thread; no other thread can
        // read the environment concurrently.
        unsafe { std::env::remove_var(key) };
    }
}

/// Prepend the hermetic git binary (via `GIT_BIN_PATH`) to `PATH` so that
/// `Command::new("git")` in test helpers resolves to the Bazel-provided
/// static binary instead of relying on system-installed git.
///
/// Safe to call multiple times — only the first call mutates `PATH`.
pub(crate) fn ensure_hermetic_git_on_path() {
    use std::path::PathBuf;
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        if let Ok(git_bin) = std::env::var("GIT_BIN_PATH") {
            let p = PathBuf::from(&git_bin);
            let p = if p.is_relative() {
                std::env::current_dir().unwrap().join(&p)
            } else {
                p
            };
            if let Some(dir) = p.parent() {
                let cur = std::env::var("PATH").unwrap_or_default();
                unsafe {
                    std::env::set_var("PATH", format!("{}:{}", dir.display(), cur));
                    // git-minimal spawns subcommands (`git stash` → `git
                    // update-index`) through its exec path, which is baked to
                    // a build-machine prefix. Helpers live next to the binary,
                    // so point the exec path there. Skip the host-fallback
                    // wrapper: host git must keep its own exec path.
                    if p.file_name().is_some_and(|name| name == "git") {
                        std::env::set_var("GIT_EXEC_PATH", dir);
                    }
                }
            }
        }
    });
}
