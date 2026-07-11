"""Tests for kenzo.config.settings: load/save/set_value round trips against
an isolated KENZO_HOME, default config shape, and ${VAR} env-placeholder
resolution.
"""

from __future__ import annotations

import pytest

from kenzo.config.settings import (
    KenzoConfig,
    PermissionsConfig,
    ProviderConfig,
    RoutingConfig,
    config_path,
    ensure_config_exists,
    load_config,
    save_config,
    set_value,
)
from kenzo.config.settings import kenzo_home as kenzo_home_fn

_ROUTING_KEYS = {
    "coding",
    "debugging",
    "vision",
    "review",
    "planning",
    "research",
    "general",
}


@pytest.fixture(autouse=True)
def _clear_provider_env_vars(monkeypatch):
    """Ensure the default config's ${VAR} placeholders resolve to "" unless
    a test explicitly sets them, regardless of the host machine's env."""
    for var in ("NVIDIA_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_kenzo_home_and_config_path_respect_env_var(kenzo_home):
    assert kenzo_home_fn() == kenzo_home
    assert config_path() == kenzo_home / "config.yaml"


def test_ensure_config_exists_creates_file_once(kenzo_home):
    path = ensure_config_exists()
    assert path == kenzo_home / "config.yaml"
    assert path.exists()

    # Mutate the on-disk file, then call again: must not be clobbered.
    marker = path.read_text(encoding="utf-8") + "\n# marker\n"
    path.write_text(marker, encoding="utf-8")

    ensure_config_exists()
    assert path.read_text(encoding="utf-8") == marker


def test_default_config_has_expected_providers_and_routing_keys(kenzo_home):
    config = load_config()

    assert isinstance(config, KenzoConfig)
    assert set(config.providers.keys()) == {"nemotron", "gemini", "openai_compatible"}
    assert set(RoutingConfig.model_fields.keys()) == _ROUTING_KEYS

    assert config.routing.coding == "nemotron"
    assert config.routing.debugging == "nemotron"
    assert config.routing.vision == "gemini"
    assert config.routing.review == "nemotron"
    assert config.routing.planning == "nemotron"
    assert config.routing.research == "nemotron"
    assert config.routing.general == "nemotron"


def test_default_config_permissions_and_repo_path(kenzo_home):
    config = load_config()

    assert config.permissions.allow_shell_exec is False
    assert config.permissions.allow_write is True
    assert config.permissions.allow_git_write is False
    assert "git" in config.permissions.command_allowlist
    assert config.default_repo_path is None


def test_env_var_placeholders_resolve_when_set(kenzo_home, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "secret-nemotron-key")
    monkeypatch.setenv("GEMINI_API_KEY", "secret-gemini-key")

    config = load_config()

    assert config.providers["nemotron"].api_key == "secret-nemotron-key"
    assert config.providers["gemini"].api_key == "secret-gemini-key"


def test_env_var_placeholders_resolve_to_empty_when_unset(kenzo_home):
    config = load_config()

    assert config.providers["nemotron"].api_key == ""
    assert config.providers["gemini"].api_key == ""
    assert config.providers["openai_compatible"].api_key == ""


def test_save_and_reload_round_trip(kenzo_home, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "resolved-secret")

    config = load_config()
    config.routing.coding = "gemini"
    config.permissions.allow_shell_exec = True
    save_config(config)

    reloaded = load_config()
    assert reloaded.routing.coding == "gemini"
    assert reloaded.permissions.allow_shell_exec is True
    # save_config persists already-resolved values (no ${VAR} placeholders
    # survive a save), so the resolved secret round-trips as a literal.
    assert reloaded.providers["nemotron"].api_key == "resolved-secret"


def test_save_config_returns_config_path(kenzo_home):
    config = load_config()
    returned_path = save_config(config)
    assert returned_path == kenzo_home / "config.yaml"


def test_set_value_updates_single_key_and_preserves_others(kenzo_home, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "still-a-placeholder-secret")

    updated = set_value("routing.coding", "gemini")

    assert updated.routing.coding == "gemini"
    assert updated.routing.debugging == "nemotron"  # untouched
    # set_value edits the on-disk YAML by dotted key only, so the ${VAR}
    # placeholder for unrelated providers is preserved (still resolves).
    assert updated.providers["nemotron"].api_key == "still-a-placeholder-secret"


def test_set_value_sets_nested_provider_key_literally(kenzo_home):
    updated = set_value("providers.gemini.api_key", "literal-key-value")
    assert updated.providers["gemini"].api_key == "literal-key-value"


def test_set_value_parses_boolean_literals(kenzo_home):
    updated = set_value("permissions.allow_write", "false")
    assert updated.permissions.allow_write is False

    updated_again = set_value("permissions.allow_write", "true")
    assert updated_again.permissions.allow_write is True


def test_provider_lookup_raises_keyerror_for_unknown_provider(kenzo_home):
    config = load_config()
    with pytest.raises(KeyError):
        config.provider("does-not-exist")


def test_kenzoconfig_defaults_when_constructed_directly():
    """KenzoConfig() with no args should still produce sane nested defaults,
    independent of the on-disk YAML (useful for unit tests elsewhere that
    build a config in-memory)."""
    config = KenzoConfig()
    assert config.providers == {}
    assert isinstance(config.routing, RoutingConfig)
    assert isinstance(config.permissions, PermissionsConfig)
    assert config.routing.general == "nemotron"


def test_providerconfig_defaults():
    provider = ProviderConfig()
    assert provider.api_key == ""
    assert provider.base_url is None
    assert provider.default_model == ""
