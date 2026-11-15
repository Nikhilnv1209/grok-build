//! `FEATURES` is the source of truth and the operator tables are hand-maintained mirrors with no compile-time check of their own.
//! This test is that check.
//!
//! The operator docs live outside this checkout in the monorepo, so the check
//! degrades to a no-op when they are absent instead of failing to compile.

use xai_grok_shell::agent::config::FEATURES;

fn operator_doc(name: &str) -> Option<String> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("docs/internal")
        .join(name);
    std::fs::read_to_string(path).ok()
}

#[test]
fn every_registered_feature_reaches_the_operator() {
    let (Some(enterprise), Some(env_vars)) = (
        operator_doc("25-enterprise.md"),
        operator_doc("22-environment-variables.md"),
    ) else {
        eprintln!("operator docs are not part of this checkout; skipping the mirror check");
        return;
    };

    for spec in FEATURES {
        assert!(
            enterprise.contains(&format!("`{}`", spec.key)),
            "{} has no row in the 25-enterprise.md pinning table",
            spec.key,
        );
        assert!(
            env_vars.contains(&format!("`{}`", spec.env)),
            "{} is undocumented in 22-environment-variables.md",
            spec.env,
        );
    }
}
