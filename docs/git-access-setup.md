# Pushing to this fork: credentials, expiries, and recovery

This document explains how pushes to this fork authenticate, why a failing
push usually means one specific thing, and how to recover — with or without
help from an AI agent. It is intentionally generic: replace the placeholders
(`<...>`) with the concrete values at fix time.

## 1. How authentication is set up here

| Thing | Setup |
|---|---|
| Source repo | kept as the `upstream` remote |
| Fork | kept as the `origin` remote: `https://<FORK_ACCOUNT>@github.com/<FORK_ACCOUNT>/<REPO>.git` — the username in the URL is load-bearing; it pins git to the right credential |
| Fork account's token | stored in a dedicated file `<CREDENTIAL_FILE>` (e.g. `~/.git-credentials-<FORK_ACCOUNT>`), mode `600` |
| Which helper reads it | set **per repo** in `.git/config`: `credential.helper = store --file=<CREDENTIAL_FILE>` — this file is never committed or pushed |
| Commit identity here | repo-local `user.name` / `user.email` matching the fork account |

## 2. Why no other repo can accidentally use this account

- Other repos use the machine's global credential helper (a different store
  file) and have remote URLs without `<FORK_ACCOUNT>@` in them.
- The two accounts never live in the same credential file, and only this repo
  reads `<CREDENTIAL_FILE>`.
- A fresh clone of this repo elsewhere loses the per-repo helper (`.git/config`
  doesn't travel) — intentional isolation, not a bug.

## 3. When a push fails — what it actually means

If `git push` fails here with:

```
remote: Invalid username or token.
fatal: Authentication failed for 'https://<FORK_ACCOUNT>@github.com/...'
```

or an HTTP 403 — the token has **expired or been revoked**. Nothing else on
the machine is broken: other repos keep working because they use a different
credential. This is the expected failure mode, not a misconfiguration.

Quick diagnosis:

```bash
git ls-remote origin           # fails with the same auth error → token problem
git config credential.helper   # in this repo: should still print the file-based helper
```

If `credential.helper` no longer points at the file (e.g. after a fresh
clone), the repo-local config was lost. Restore it:

```bash
git remote set-url origin https://<FORK_ACCOUNT>@github.com/<FORK_ACCOUNT>/<REPO>.git
git config credential.helper 'store --file=<CREDENTIAL_FILE>'
```

## 4. Fix: swap in a new token

1. On the **fork account**: Settings → Developer settings → Personal access
   tokens → Tokens (classic) → Generate new token (classic), scope `repo`,
   any expiry.
2. Put the new token in the dedicated file:

   ```bash
   printf 'https://<FORK_ACCOUNT>:%s@github.com\n' NEW_TOKEN > <CREDENTIAL_FILE>
   chmod 600 <CREDENTIAL_FILE>
   ```

3. Verify: `git ls-remote origin` succeeds. No prompts, nothing else to change.
   The old token can then be revoked from the account settings.

## 5. If asking an AI agent for help — what it needs from you

To fix this without digging through history, an agent only needs:

1. Which repo is failing (name/path) and **which remote** (`upstream` or `origin`
   `git push` targets).
2. The **fork account's username**.
3. The exact error output from the failed command.
4. Whether the command in section 3 shows the per-repo helper or not.
5. Either a **newly generated token** (see section 4) you can paste, or
   permission to switch this repo to SSH (section 6).

With those five items, the recovery is mechanical — no other context required.

## 6. Optional: replace the token with an SSH key (no expiry)

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_fork
cat ~/.ssh/id_ed25519_fork.pub   # paste into the fork account: Settings → SSH and GPG keys
```

Add to `~/.ssh/config`:

```
Host github-fork
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_fork
```

Then: `git remote set-url origin git@github-fork:<FORK_ACCOUNT>/<REPO>.git`.
Pushes then authenticate by key; the token file is ignored and never expires.

## 7. Staying in sync with the source repo

`main` tracks `upstream/main`. To fold upstream changes in:

```bash
git fetch upstream
git checkout main && git merge upstream/main
```