#!/usr/bin/env bash
# Get the Modal CLI working in a Claude Code on the web session, then prove it can reach the workspace.
# Usage: bash .claude/skills/modal/check.sh        (idempotent; safe to rerun)
set -u

# The session's outbound HTTPS goes through an HTTP proxy (HTTPS_PROXY). Modal's gRPC client only honours it
# when python-socks is installed; without it every command fails with "Could not connect to the Modal server".
if ! command -v modal >/dev/null 2>&1 || ! "$(dirname "$(readlink -f "$(command -v modal)")")/python" -c "import python_socks" 2>/dev/null; then
    echo "installing modal[api-proxy-support] ..."
    if command -v uv >/dev/null 2>&1; then
        uv tool install --force 'modal[api-proxy-support]' >/dev/null 2>&1 || { echo "uv tool install failed"; exit 1; }
    else
        pip install -q 'modal[api-proxy-support]' || { echo "pip install failed"; exit 1; }
    fi
fi
modal --version

if [ ! -f ~/.modal.toml ] && [ -z "${MODAL_TOKEN_ID:-}" ]; then
    echo "NO CREDENTIALS: no ~/.modal.toml and no MODAL_TOKEN_ID/MODAL_TOKEN_SECRET. Ask the user to add them to"
    echo "the environment's variables (https://code.claude.com/docs/en/claude-code-on-the-web)."
    exit 2
fi

if ! timeout 90 modal token info 2>&1 | grep -E '^(Workspace|User):'; then
    echo "modal token info failed. Check the proxy: curl -sS \"\$HTTPS_PROXY/__agentproxy/status\""
    echo "(a 403 for api.modal.com in recentRelayFailures means the network policy blocks Modal; report it, don't route around it)"
    exit 3
fi

echo "--- volumes this repo needs (laya-hf-cache, laya-datasets, laya-checkpoints):"
timeout 90 modal volume list 2>&1 | grep -oE 'laya-(hf-cache|datasets|checkpoints)' | sort -u
