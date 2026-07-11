"""Translation layer between Kenzo's provider config and litellm's model
addressing / call-kwarg conventions.

`kenzo.config.settings.ProviderConfig` stores a provider-agnostic
`default_model` string plus connection details (`api_key`, `base_url`).
litellm, on the other hand, expects a single fully-qualified model id of the
form "<litellm-provider>/<model-name>" (e.g. "gemini/gemini-1.5-pro",
"openai/nvidia/nemotron-4-340b-instruct"). The functions here bridge that gap
so the router doesn't need to know provider-specific quirks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kenzo.config.settings import ProviderConfig


def resolve_model_string(provider_name: str, provider_config: ProviderConfig) -> str:
    """Build the litellm-style "<provider>/<model>" id for a configured provider.

    - If `default_model` already contains a slash, it's treated as an
      already-fully-qualified litellm model id and returned unchanged.
    - If `provider_name` is "gemini", the model is prefixed with "gemini/".
    - Every other provider (e.g. "nemotron", "openai_compatible", or any
      custom provider name) is assumed to be reachable via an
      OpenAI-compatible endpoint and is prefixed with "openai/", per litellm
      conventions for custom/self-hosted OpenAI-compatible providers.
    """
    default_model = provider_config.default_model

    if "/" in default_model:
        return default_model

    if provider_name == "gemini":
        return f"gemini/{default_model}"

    return f"openai/{default_model}"


def build_completion_kwargs(provider_config: ProviderConfig) -> dict[str, Any]:
    """Build the litellm.completion() kwargs derived from a ProviderConfig.

    Only truthy values are included so litellm falls back to its own
    defaults (e.g. env-var based auth) when a field is unset.
    """
    candidate = {
        "api_key": provider_config.api_key,
        "api_base": provider_config.base_url,
    }
    return {key: value for key, value in candidate.items() if value}
