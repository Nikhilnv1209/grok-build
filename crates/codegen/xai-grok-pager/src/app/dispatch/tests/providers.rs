//! Tests for the provider dialog dispatch: session-less hosting on welcome
//! and the credential-gate lift after a successful connect.

use super::*;

#[test]
fn open_providers_without_session_hosts_welcome_modal() {
    let mut app = test_app();
    // Fresh launch: no agents, welcome view.
    assert!(app.agents.is_empty());
    let effects = dispatch(Action::OpenProviders, &mut app);
    assert!(effects.is_empty());
    assert!(
        matches!(
            app.welcome_modal,
            Some(crate::views::modal::ActiveModal::Providers { .. })
        ),
        "dialog must open at app level before any session exists"
    );
}

#[test]
#[serial_test::serial(GROK_HOME)]
fn submit_connect_key_lifts_credential_gate() {
    // store_key writes ~/.grok/auth.json — redirect GROK_HOME to a tempdir.
    let home = tempfile::tempdir().unwrap();
    let _guard = crate::test_util::EnvVarGuard::set("GROK_HOME", home.path());
    let mut app = test_app();
    app.auth_state = crate::app::app_view::AuthState::Done;
    app.has_any_credential = false;
    app.has_external_auth_provider = false;

    let effects = dispatch(
        Action::SubmitConnectKey {
            provider: "deepseek".into(),
            key: "sk-test-gate-lift".into(),
        },
        &mut app,
    );

    assert!(app.has_any_credential, "gate must lift after connect");
    assert!(app.has_external_auth_provider);
    assert!(
        effects
            .iter()
            .any(|e| matches!(e, crate::app::actions::Effect::ReloadAgentModels)),
        "connect must still schedule the live catalog reload"
    );
}

#[test]
#[serial_test::serial(GROK_HOME)]
fn connect_from_welcome_opens_and_submits_session_less() {
    let home = tempfile::tempdir().unwrap();
    let _guard = crate::test_util::EnvVarGuard::set("GROK_HOME", home.path());
    let mut app = test_app();
    app.auth_state = crate::app::app_view::AuthState::Done;
    app.has_any_credential = false;

    dispatch(Action::ConnectProvider("opencode".into()), &mut app);
    assert!(matches!(
        app.welcome_modal,
        Some(crate::views::modal::ActiveModal::ConnectProvider { .. })
    ));

    // Esc closes the key modal without touching credentials.
    let esc = crossterm::event::KeyEvent::new(
        crossterm::event::KeyCode::Esc,
        crossterm::event::KeyModifiers::NONE,
    );
    app.handle_input(&crossterm::event::Event::Key(esc));
    assert!(app.welcome_modal.is_none());
    assert!(!app.has_any_credential, "cancelled connect keeps the gate");
}
