"""
E.D.I.T.H. backend checks — LLM integration hardening.

Run all offline checks:      python test_main.py
Probe the real provider:     python test_main.py --live
List models for this key:    python test_main.py --models

The offline suite never touches the network: the provider is stubbed so every
failure mode (401, 429, timeout, garbage body, open breaker, missing key) can
be exercised deterministically.
"""

import asyncio
import json
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager, redirect_stdout
from io import StringIO

import main

EXPECTED_ERROR = {
    "is_justified": False,
    "reason": "API Error: Defaulting to Zero-Trust Block",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def reset_breaker() -> None:
    main._llm_failures = 0
    main._llm_open_until = 0.0


@contextmanager
def provider_stub(stub=None, key="test-key"):
    """Swap the provider and the API key for the duration of a check."""
    original_provider = main._call_provider
    original_key = main.API_KEY
    main._call_provider = stub or (lambda *a, **k: '{"ok": true}')
    main.API_KEY = key
    reset_breaker()
    try:
        yield
    finally:
        main._call_provider = original_provider
        main.API_KEY = original_key
        reset_breaker()


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 1. Markdown-tolerant JSON parsing
# ---------------------------------------------------------------------------
def test_plain_json():
    assert main.extract_json('{"is_justified": true}') == {"is_justified": True}


def test_json_fence_with_language():
    raw = '```json\n{"is_justified": false, "reason": "not core"}\n```'
    assert main.extract_json(raw) == {"is_justified": False, "reason": "not core"}


def test_json_fence_uppercase_and_bare():
    assert main.extract_json('```JSON\n{"a": 1}\n```') == {"a": 1}
    assert main.extract_json('```\n{"a": 2}\n```') == {"a": 2}


def test_prose_around_fence():
    raw = (
        "Sure! Here is the verdict:\n"
        '```json\n{"is_justified": true, "reason": "Camera scans QR codes."}\n```\n'
        "Let me know if you need anything else."
    )
    parsed = main.extract_json(raw)
    assert parsed["is_justified"] is True
    assert parsed["reason"] == "Camera scans QR codes."


def test_trailing_comma_tolerated():
    assert main.extract_json('{"is_justified": true, "reason": "core",}') == {
        "is_justified": True,
        "reason": "core",
    }


def test_garbage_returns_none():
    assert main.extract_json("") is None
    assert main.extract_json(None) is None
    assert main.extract_json("the model refused to answer") is None
    assert main.extract_json("[1, 2, 3]") is None


# ---------------------------------------------------------------------------
# 2. Missing / invalid key policy
# ---------------------------------------------------------------------------
def test_missing_key_fails_closed():
    buffer = StringIO()
    with provider_stub(key=""):
        with redirect_stdout(buffer):
            raw = run(main.call_llm("sys", "user"))
    assert json.loads(raw) == EXPECTED_ERROR
    assert "no usable API key configured" in buffer.getvalue()


def test_401_logs_and_fails_closed():
    def unauthorized(*_a, **_k):
        raise urllib.error.HTTPError("https://api.example", 401, "Unauthorized", None, None)

    buffer = StringIO()
    with provider_stub(unauthorized):
        with redirect_stdout(buffer):
            raw = run(main.call_llm("sys", "user"))
    assert json.loads(raw) == EXPECTED_ERROR
    assert "API key rejected" in buffer.getvalue()


def test_429_logs_rate_limit():
    def limited(*_a, **_k):
        raise urllib.error.HTTPError("https://api.example", 429, "Too Many Requests", None, None)

    buffer = StringIO()
    with provider_stub(limited):
        with redirect_stdout(buffer):
            raw = run(main.call_llm("sys", "user"))
    assert json.loads(raw) == EXPECTED_ERROR
    assert "rate limited" in buffer.getvalue()


def test_timeout_and_network_errors_fail_closed():
    def raiser(error):
        def boom(*_a, **_k):
            raise error

        return boom

    for error in (urllib.error.URLError("connection refused"), TimeoutError("timed out")):
        with provider_stub(raiser(error)):
            raw = run(main.call_llm("sys", "user"))
        assert json.loads(raw) == EXPECTED_ERROR


def test_empty_completion_fails_closed():
    with provider_stub(lambda *a, **k: "   "):
        raw = run(main.call_llm("sys", "user"))
    assert json.loads(raw) == EXPECTED_ERROR


def test_key_never_appears_in_logs():
    secret = "sk-live-TOPSECRET-999"

    def leaky(*_a, **_k):
        raise urllib.error.HTTPError(
            "https://api.example?key=" + secret, 500, "boom " + secret, None, None
        )

    buffer = StringIO()
    with provider_stub(leaky, key=secret):
        with redirect_stdout(buffer):
            run(main.call_llm("sys", "user"))
    assert secret not in buffer.getvalue()
    assert "[redacted]" in buffer.getvalue()

    # The URL-query guard also works for keys that are not the configured one.
    assert "another-secret" not in main._redact_secrets(
        "https://api.example?key=another-secret&x=1"
    )


# ---------------------------------------------------------------------------
# 3. Circuit breaker
# ---------------------------------------------------------------------------
def test_breaker_short_circuits_without_calling_provider():
    calls = {"count": 0}

    def should_not_run(*_a, **_k):
        calls["count"] += 1
        return '{"is_justified": true, "reason": "should not happen"}'

    with provider_stub(should_not_run):
        main._llm_open_until = time.monotonic() + 60
        raw = run(main.call_llm("sys", "user"))
        assert calls["count"] == 0
        assert json.loads(raw) == EXPECTED_ERROR


def test_success_resets_failure_counter():
    with provider_stub(lambda *a, **k: '{"is_justified": true, "reason": "core"}'):
        main._llm_failures = 1
        run(main.call_llm("sys", "user"))
        assert main._llm_failures == 0


def test_two_failures_open_the_breaker():
    def boom(*_a, **_k):
        raise urllib.error.URLError("down")

    with provider_stub(boom):
        run(main.call_llm("sys", "user"))
        run(main.call_llm("sys", "user"))
        assert main.llm_breaker_open() is True


# ---------------------------------------------------------------------------
# 4. Strict prompt + verdict handling
# ---------------------------------------------------------------------------
def test_strict_system_prompt_is_sent():
    seen = {}

    def stub(_key, system_prompt, user_prompt):
        seen["system"] = system_prompt
        seen["user"] = user_prompt
        return '{"is_justified": true, "reason": "Camera is required to scan payment QR codes."}'

    with provider_stub(stub):
        verdict = run(main.analyse_request("Paytm", "camera"))

    assert "enterprise Zero-Trust Firewall" in seen["system"]
    assert "App 'Paytm' is requesting 'camera'." in seen["system"]
    assert "Do not assume future features." in seen["system"]
    assert '"is_justified": boolean' in seen["system"]
    assert verdict == {
        "is_justified": True,
        "reason": "Camera is required to scan payment QR codes.",
        "source": f"llm:{main.LLM_PROVIDER}",
    }


def test_fenced_reply_is_accepted():
    fenced = '```json\n{"is_justified": false, "reason": "Not core."}\n```'
    with provider_stub(lambda *a, **k: fenced):
        verdict = run(main.analyse_request("Calculator", "contacts"))
    assert verdict["is_justified"] is False
    assert verdict["reason"] == "Not core."
    assert verdict["source"].startswith("llm:")


def test_string_boolean_is_coerced():
    with provider_stub(lambda *a, **k: '{"is_justified": "false", "reason": "Not core."}'):
        assert run(main.analyse_request("WeatherApp", "contacts"))["is_justified"] is False

    with provider_stub(lambda *a, **k: '{"is_justified": "true", "reason": "Core feature."}'):
        assert run(main.analyse_request("Paytm", "sms"))["is_justified"] is True


def test_provider_error_becomes_fail_closed_verdict():
    def boom(*_a, **_k):
        raise urllib.error.HTTPError("https://api.example", 503, "Unavailable", None, None)

    with provider_stub(boom):
        verdict = run(main.analyse_request("Calculator", "contacts"))
    assert verdict == {
        "is_justified": False,
        "reason": "API Error: Defaulting to Zero-Trust Block",
        "source": "llm-error",
    }


def test_unparseable_completion_becomes_fail_closed_verdict():
    with provider_stub(lambda *a, **k: "I cannot answer that."):
        verdict = run(main.analyse_request("Calculator", "contacts"))
    assert verdict["is_justified"] is False
    assert verdict["reason"] == "API Error: Defaulting to Zero-Trust Block"
    assert verdict["source"] == "llm-error"


def test_missing_is_justified_becomes_fail_closed_verdict():
    with provider_stub(lambda *a, **k: '{"reason": "Core feature."}'):
        verdict = run(main.analyse_request("Paytm", "camera"))
    assert verdict["is_justified"] is False
    assert verdict["source"] == "llm-error"


def test_missing_key_blocks_every_request():
    with provider_stub(key=""):
        verdict = run(main.analyse_request("WeatherApp", "contacts"))
    assert verdict["is_justified"] is False
    assert verdict["reason"] == "API Error: Defaulting to Zero-Trust Block"


# ---------------------------------------------------------------------------
# 5. Fail-closed payload contract
# ---------------------------------------------------------------------------
def test_fail_closed_payload_shape():
    assert main.LLM_ERROR_PAYLOAD == EXPECTED_ERROR
    assert json.loads(main.LLM_ERROR_JSON) == EXPECTED_ERROR
    assert main.strip_markdown(main.LLM_ERROR_JSON) == main.LLM_ERROR_JSON


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
CHECKS = [
    test_plain_json,
    test_json_fence_with_language,
    test_json_fence_uppercase_and_bare,
    test_prose_around_fence,
    test_trailing_comma_tolerated,
    test_garbage_returns_none,
    test_missing_key_fails_closed,
    test_401_logs_and_fails_closed,
    test_429_logs_rate_limit,
    test_timeout_and_network_errors_fail_closed,
    test_empty_completion_fails_closed,
    test_key_never_appears_in_logs,
    test_breaker_short_circuits_without_calling_provider,
    test_success_resets_failure_counter,
    test_two_failures_open_the_breaker,
    test_strict_system_prompt_is_sent,
    test_fenced_reply_is_accepted,
    test_string_boolean_is_coerced,
    test_provider_error_becomes_fail_closed_verdict,
    test_unparseable_completion_becomes_fail_closed_verdict,
    test_missing_is_justified_becomes_fail_closed_verdict,
    test_missing_key_blocks_every_request,
    test_fail_closed_payload_shape,
]


def live_check() -> int:
    """One real provider probe; never prints the key."""
    print("Probing the configured provider ...")
    raw = run(main.call_llm("Reply with JSON only.", 'Return exactly {"ok": true}.'))
    if main.is_llm_error_response(raw):
        print("RESULT: provider unavailable or key rejected — requests will FAIL CLOSED.")
        return 1
    print("RESULT: provider reachable. Reply:", str(raw)[:120])
    return 0


def list_models() -> int:
    """List Gemini models this key can see (prints names only, never the key)."""
    key = main.usable_api_key()
    if key is None:
        print("No usable API key configured.")
        return 1
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(f"Model list failed: HTTP {exc.code}. The API key itself is likely invalid.")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Model list failed: {type(exc).__name__}: {exc}")
        return 1

    names = [m.get("name", "") for m in data.get("models", [])]
    generate = [
        m.get("name", "")
        for m in data.get("models", [])
        if "generateContent" in (m.get("supportedGenerationMethods") or [])
    ]
    print(f"Key is valid. {len(names)} models visible; {len(generate)} support generateContent:")
    for name in generate:
        print("  ", name.split("/", 1)[-1])
    return 0


def run_checks() -> int:
    failures = 0
    for check in CHECKS:
        try:
            check()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {check.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - report every unexpected failure
            failures += 1
            print(f"ERROR {check.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS  {check.__name__}")

    passed = len(CHECKS) - failures
    print(f"\n{passed}/{len(CHECKS)} checks passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    if "--models" in sys.argv:
        sys.exit(list_models())
    if "--live" in sys.argv:
        sys.exit(live_check())
    sys.exit(run_checks())
