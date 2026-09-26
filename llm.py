"""
llm.py
------
Responsible ONLY for:
    - connecting to the inference API
    - sending the prompt
    - retrieving the response

No safety logic and no prompt assembly lives here (see section 13).

The MVP is written against a generic HTTP inference API so it can point at
a Hugging Face Inference Endpoint, a self-hosted BioMistral-7B server
(e.g. text-generation-inference / vLLM), or any OpenAI-compatible
"/chat/completions" route — configured entirely through environment
variables, never hard-coded.
"""

import os
import requests

try:
    # streamlit is optional here only so this file can still be imported/
    # unit-tested outside of a Streamlit run (e.g. in the quick tests we
    # ran earlier). It is always available in the deployed app.
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None


class LLMError(Exception):
    """Raised for any problem talking to the inference API."""
    pass


def _get_setting(name: str, default: str = None):
    """
    Reads a setting from, in order of priority:
    1) Streamlit Cloud "Secrets" (st.secrets) — required on Streamlit
       Community Cloud, where .env files are not deployed.
    2) A local .env file / real environment variable (os.getenv) — used
       for local development.
    This lets the exact same code run unmodified locally and on the
    free Streamlit Cloud.
    """
    if st is not None:
        try:
            if name in st.secrets:
                return st.secrets[name]
        except Exception:
            pass  # no secrets.toml configured (e.g. pure local run) — fine
    return os.getenv(name, default)


DEFAULT_MAX_NEW_TOKENS = int(_get_setting("MAX_NEW_TOKENS", "250"))
DEFAULT_TEMPERATURE = float(_get_setting("TEMPERATURE", "0.3"))
DEFAULT_TIMEOUT = int(_get_setting("REQUEST_TIMEOUT", "30"))


def _get_config():
    api_url = _get_setting("BIOMISTRAL_API_URL")
    api_key = _get_setting("BIOMISTRAL_API_KEY")
    api_mode = _get_setting("BIOMISTRAL_API_MODE", "hf")  # "hf" or "openai"
    return api_url, api_key, api_mode


def _build_payload(prompt: str, max_new_tokens: int, temperature: float, api_mode: str) -> dict:
    if api_mode == "openai":
        # OpenAI-compatible chat completion route (common for self-hosted
        # vLLM / TGI servers exposing an OpenAI-style API).
        return {
            "model": os.getenv("BIOMISTRAL_MODEL_NAME", "BioMistral-7B"),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
    # Default: Hugging Face style text-generation inference API.
    return {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "return_full_text": False,
        },
    }


def _extract_text(data, api_mode: str) -> str:
    try:
        if api_mode == "openai":
            return data["choices"][0]["message"]["content"]

        # Hugging Face inference API can return either a list of dicts
        # or a single dict depending on the model/server.
        if isinstance(data, list) and data:
            item = data[0]
            if isinstance(item, dict):
                return item.get("generated_text", "")
        if isinstance(data, dict):
            if "generated_text" in data:
                return data["generated_text"]
            if "error" in data:
                raise LLMError(f"Model error: {data['error']}")
        return ""
    except (KeyError, IndexError, TypeError):
        return ""


def query_biomistral(
    prompt: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """
    Sends `prompt` to the configured inference API and returns the
    generated text. Raises LLMError on any failure (API down, timeout,
    connection issue, empty response) so the caller (app.py) can show a
    single, friendly error message (see section 14).
    """
    api_url, api_key, api_mode = _get_config()

    if not api_url or not api_key:
        raise LLMError(
            "The model API is not configured. Set BIOMISTRAL_API_URL and "
            "BIOMISTRAL_API_KEY (see .env.example)."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = _build_payload(prompt, max_new_tokens, temperature, api_mode)

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=DEFAULT_TIMEOUT)
    except requests.exceptions.Timeout:
        raise LLMError("The request to the model timed out. Please try again later.")
    except requests.exceptions.ConnectionError:
        raise LLMError("Unable to contact the model. Please check your connection.")
    except requests.exceptions.RequestException as exc:
        raise LLMError(f"Unexpected error contacting the model: {exc}")

    if response.status_code != 200:
        raise LLMError(
            f"The model is currently unavailable (status {response.status_code}). "
            "Please try again later."
        )

    try:
        data = response.json()
    except ValueError:
        raise LLMError("Received an invalid response from the model.")

    text = _extract_text(data, api_mode)

    if not text or not text.strip():
        raise LLMError("The model returned an empty response. Please try again.")

    return text.strip()
