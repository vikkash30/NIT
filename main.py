"""
E.D.I.T.H. - Continuous Privacy & Permission Management System
Central Backend API

Run with:
    uvicorn main:app --reload
    http://127.0.0.1:8000/docs

===============================================================================
 HOW TO ADD YOUR LLM API KEY
===============================================================================
 Put the key in a .env file next to this module (never hard-code it):

     GEMINI_API_KEY_1=your_key_here
     GEMINI_API_KEY_2=optional_failover_key
     LLM_PROVIDER=gemini          # or openai   (default: gemini)
     LLM_MODEL=gemini-3.5-flash-lite

 If the key is missing, rejected, rate-limited, unparseable, or the provider is
 unreachable, the backend fails closed: every AI verdict becomes the zero-trust
 block payload
 {"is_justified": false, "reason": "API Error: Defaulting to Zero-Trust Block"}
 instead of crashing or guessing.

 A circuit breaker watches the provider: after LLM_FAILURE_THRESHOLD consecutive
 failures the LLM is skipped for LLM_COOLDOWN_SECONDS, so a dead/expired key can
 never make a live demo hang. A warm-up probe runs once at startup.

 Verify the integration with:
     python test_main.py            # offline checks (stubbed provider)
     python test_main.py --live     # one real provider round-trip
     python test_main.py --models   # list models available to this key
===============================================================================
"""

import asyncio
import itertools
import json
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# >>>>>>>>>>>>>>>>>>>>>>>>   API KEYS & CONFIGURATION   <<<<<<<<<<<<<<<<<<<<<<<
# =============================================================================
# Load API keys from environment variables (.env)
API_KEYS = [
    key for key in [
        os.getenv("GEMINI_API_KEY_1"),
        os.getenv("GEMINI_API_KEY_2"),
        os.getenv("GEMINI_API_KEY_3"),
    ] if key
]
API_KEY = os.getenv("GEMINI_API_KEY_1") or (API_KEYS[0] if API_KEYS else os.getenv("API_KEY", ""))
# =============================================================================

# Provider configuration ---------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL") or (
    # 2.0-flash and 2.5-flash are retired for new keys; 3.5-flash-lite answers
    # the strict JSON verdict in ~1.5s.
    "gemini-3.5-flash-lite" if LLM_PROVIDER == "gemini" else "gpt-4o-mini"
)
LLM_TIMEOUT_SECONDS = 15
LLM_FAILURE_THRESHOLD = 2      # consecutive failures before the breaker opens
LLM_COOLDOWN_SECONDS = 180     # how long the breaker stays open

# Placeholder strings that must never be sent to a provider.
PLACEHOLDER_KEYS = {
    "PASTE_YOUR_API_KEY_HERE",
    "KEY_1_HERE",
    "KEY_2_HERE",
    "KEY_3_HERE",
    "",
}


def usable_api_key() -> Optional[str]:
    """Return the configured key, or None when it is blank/placeholder."""
    key = (API_KEY or "").strip()
    return None if key in PLACEHOLDER_KEYS else key


# ---------------------------------------------------------------------------
# Fail-closed LLM verdict
# ---------------------------------------------------------------------------
# If the provider cannot be trusted (missing key, 401/403, 429, timeout,
# malformed body) we never guess and never crash: the firewall answers with
# this fixed zero-trust verdict instead.
LLM_ERROR_REASON = "API Error: Defaulting to Zero-Trust Block"
LLM_ERROR_PAYLOAD = {"is_justified": False, "reason": LLM_ERROR_REASON}
LLM_ERROR_JSON = json.dumps(LLM_ERROR_PAYLOAD)

# Strict prompt: one resource, current functionality only, no speculation.
STRICT_AUDIT_PROMPT = (
    "You are an enterprise Zero-Trust Firewall. "
    "App '{app_name}' is requesting '{resource}'. "
    "Does it fundamentally require this to operate? "
    'Reply strictly in JSON: {{"is_justified": boolean, "reason": "1-sentence explanation"}}. '
    "Do not assume future features."
)


def is_llm_error_response(raw: Optional[str]) -> bool:
    """True when `raw` is the fail-closed payload, not a real completion."""
    return not raw or str(raw).strip() == LLM_ERROR_JSON


# ---------------------------------------------------------------------------
# LLM circuit breaker state
# ---------------------------------------------------------------------------
_llm_lock = threading.Lock()
_llm_failures = 0
_llm_open_until = 0.0


def llm_breaker_open() -> bool:
    return time.monotonic() < _llm_open_until


def _record_llm_success() -> None:
    global _llm_failures
    with _llm_lock:
        _llm_failures = 0


def _record_llm_failure() -> None:
    global _llm_failures, _llm_open_until
    with _llm_lock:
        _llm_failures += 1
        if _llm_failures >= LLM_FAILURE_THRESHOLD:
            _llm_open_until = time.monotonic() + LLM_COOLDOWN_SECONDS
            _llm_failures = 0
            print(
                f"[E.D.I.T.H.] LLM breaker OPEN for {LLM_COOLDOWN_SECONDS}s "
                "— requests will fail closed (Zero-Trust Block)."
            )


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------
# Single shared in-memory connection. `check_same_thread=False` allows the
# connection to be used across FastAPI's threadpool workers; DB_LOCK serialises
# access so parallel sync endpoints can never interleave a write/read.
DB_CONNECTION = sqlite3.connect(":memory:", check_same_thread=False)
DB_CONNECTION.row_factory = sqlite3.Row
DB_LOCK = threading.RLock()

# ---------------------------------------------------------------------------
# Dynamic permissions state
# ---------------------------------------------------------------------------
# True  -> the app is currently permitted to use the resource
# False -> E.D.I.T.H. has revoked (or never granted) that permission
PERMISSIONS_STATE: dict[str, dict[str, bool]] = {
    # Wallet: needs SMS for OTPs, camera for QR, contacts to pay people.
    "Paytm": {
        "location": True,
        "contacts": True,
        "microphone": True,
        "sms": True,
        "camera": True,
        "files": False,
        "calendar": False,
        "bluetooth": False,
        "storage": True,
        "network": True,
    },
    # Second wallet: identical default permissions to Paytm.
    "YPay": {
        "location": True,
        "contacts": True,
        "microphone": True,
        "sms": True,
        "camera": True,
        "files": False,
        "calendar": False,
        "bluetooth": False,
        "storage": True,
        "network": True,
    },
    # Utility: only what a forecast genuinely requires.
    "WeatherApp": {
        "location": True,
        "contacts": False,
        "microphone": False,
        "sms": False,
        "camera": False,
        "files": False,
        "calendar": False,
        "bluetooth": False,
        "storage": False,
        "network": True,
    },
    # System tool: voice input only — everything invasive stays off.
    "Calculator": {
        "location": False,
        "contacts": False,
        "microphone": True,
        "sms": False,
        "camera": False,
        "files": False,
        "calendar": False,
        "bluetooth": False,
        "storage": True,
        "network": False,
    },
}

# Every resource identifier the policy engine understands. Requests for
# anything else are rejected with 400 instead of silently creating a key.
KNOWN_RESOURCES: set[str] = {
    resource for permissions in PERMISSIONS_STATE.values() for resource in permissions
}

# Lowercase index used to make every lookup case-insensitive.
APP_NAME_INDEX: dict[str, str] = {name.lower(): name for name in PERMISSIONS_STATE}

# Trust score history (for the frontend line graph).
# Every time an app's score changes, the new score is appended to its list.
TRUST_HISTORY_LIMIT = 60
trust_history: dict[str, list[int]] = {
    "paytm": [100],
    "ypay": [100],
    "weatherapp": [100],
    "calculator": [100],
}

ADMIN_BLOCK_PENALTY = 7   # admin clicked "No" in the interception modal

# Resources each app could plausibly justify. Used only as the offline
# fallback for LLM analysis (and by the audit) when no API key is configured.
PLAUSIBLE_RESOURCES: dict[str, set] = {
    "Paytm": {"location", "contacts", "microphone", "sms", "camera", "storage", "network"},
    "YPay": {"location", "contacts", "microphone", "sms", "camera", "storage", "network"},
    "WeatherApp": {"location", "network", "storage"},
    "Calculator": {"microphone", "storage", "network"},
}

# ---------------------------------------------------------------------------
# Heuristic policy
# ---------------------------------------------------------------------------
# Velocity Tracking: rolling window of request timestamps (monotonic clock).
VELOCITY_WINDOW_SECONDS = 10
VELOCITY_MAX_REQUESTS = 5
VELOCITY_PENALTY = 40     # burst / data-scraping behaviour
REQUEST_HISTORY: dict[str, deque] = defaultdict(deque)

# ---------------------------------------------------------------------------
# Real-time interception system
# ---------------------------------------------------------------------------
# request_id -> pending request record, awaiting a human verdict from the
# E.D.I.T.H. dashboard. Only requests for resources that are currently OFF are
# escalated here.
pending_requests: dict[str, dict] = {}

INTERCEPT_TIMEOUT_SECONDS = 30   # give the admin this long to respond
INTERCEPT_POLL_SECONDS = 1       # how often the waiting coroutine re-checks
# Verdict applied when nobody answers in time.
INTERCEPT_TIMEOUT_VERDICT = "Allowed"
PENDING_MAX_AGE_SECONDS = 300    # safety sweep for abandoned requests

# Notification hub: consumed by GET /get_notifications.
notifications: list[dict] = []
NOTIFICATION_LIMIT = 100
NOTIFICATION_SEQ = itertools.count(1)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class AccessRequest(BaseModel):
    app_name: str
    requested_resource: str


class ScoreUpdate(BaseModel):
    app_name: str
    trust_score: int = Field(..., ge=0, le=100)


class PermissionToggle(BaseModel):
    app_name: str
    resource: str
    state: bool


class ResolveRequest(BaseModel):
    request_id: str
    action: str
    # Optional time-limit for "Allowed" verdicts. 0 / omitted => permanent.
    duration: int = 0


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def init_db() -> None:
    """Create tables and seed the default applications."""
    with DB_LOCK:
        cursor = DB_CONNECTION.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS Applications (
                app_name    TEXT PRIMARY KEY,
                trust_score INTEGER DEFAULT 100
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS AccessLogs (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                app_name           TEXT NOT NULL,
                requested_resource TEXT NOT NULL,
                timestamp          TEXT NOT NULL,
                status             TEXT NOT NULL,
                ai_reason          TEXT DEFAULT ''
            )
            """
        )
        cursor.executemany(
            "INSERT OR IGNORE INTO Applications (app_name, trust_score) VALUES (?, ?)",
            [("Paytm", 100), ("YPay", 100), ("WeatherApp", 100), ("Calculator", 100)],
        )
        DB_CONNECTION.commit()


init_db()


def get_trust_score(app_name: str) -> Optional[int]:
    with DB_LOCK:
        cursor = DB_CONNECTION.cursor()
        cursor.execute(
            "SELECT trust_score FROM Applications WHERE app_name = ?", (app_name,)
        )
        row = cursor.fetchone()
        return row["trust_score"] if row else None


def record_trust_history(app_name: str, trust_score: int) -> None:
    """Append a new score to the app's history (skipping no-op writes)."""
    history = trust_history.setdefault(app_name.strip().lower(), [])
    if not history or history[-1] != trust_score:
        history.append(trust_score)
        del history[:-TRUST_HISTORY_LIMIT]


def set_trust_score(app_name: str, trust_score: int) -> None:
    with DB_LOCK:
        cursor = DB_CONNECTION.cursor()
        cursor.execute(
            "UPDATE Applications SET trust_score = ? WHERE app_name = ?",
            (trust_score, app_name),
        )
        DB_CONNECTION.commit()
    record_trust_history(app_name, trust_score)


def write_log(
    app_name: str, resource: str, status: str, ai_reason: str = ""
) -> str:
    """Persist an access decision (with the AI's reason) and return the timestamp."""
    timestamp = datetime.now(timezone.utc).isoformat()
    with DB_LOCK:
        cursor = DB_CONNECTION.cursor()
        cursor.execute(
            """
            INSERT INTO AccessLogs
                (app_name, requested_resource, timestamp, status, ai_reason)
            VALUES (?, ?, ?, ?, ?)
            """,
            (app_name, resource, timestamp, status, ai_reason),
        )
        DB_CONNECTION.commit()
    return timestamp


def push_notification(app_name: str, resource: str, points: int) -> dict:
    """Record an admin denial in the notification hub."""
    entry = {
        "id": next(NOTIFICATION_SEQ),
        "app_name": app_name,
        "resource": resource,
        "points": points,
        "message": (
            f"{app_name} requested {resource}. You denied it. "
            f"Trust Score reduced by {points} points."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    notifications.append(entry)
    del notifications[:-NOTIFICATION_LIMIT]
    return entry


def apply_velocity_tracking(app_name: str) -> list[str]:
    """Deduct points when an app exceeds the request rate limit."""
    now = time.monotonic()
    history = REQUEST_HISTORY[app_name]
    history.append(now)

    while history and now - history[0] > VELOCITY_WINDOW_SECONDS:
        history.popleft()

    if len(history) <= VELOCITY_MAX_REQUESTS:
        return []

    current = get_trust_score(app_name) or 0
    set_trust_score(app_name, max(0, current - VELOCITY_PENALTY))
    history.clear()   # penalise one burst once, not on every request
    return [
        f"Data Scraping Attack: more than {VELOCITY_MAX_REQUESTS} requests in "
        f"{VELOCITY_WINDOW_SECONDS}s (-{VELOCITY_PENALTY})"
    ]


def finalize_request(entry: dict, status: str, resolved_by: str) -> dict:
    """Commit a verdict exactly once: log it and apply the score penalty."""
    entry["status"] = status
    entry["resolved_by"] = resolved_by
    entry["resolved_at"] = datetime.now(timezone.utc).isoformat()

    if status == "Blocked":
        current = get_trust_score(entry["app_name"]) or 0
        set_trust_score(entry["app_name"], max(0, current - ADMIN_BLOCK_PENALTY))
        if resolved_by == "admin":
            push_notification(
                entry["app_name"], entry["resource"], ADMIN_BLOCK_PENALTY
            )

    write_log(
        entry["app_name"],
        entry["resource"],
        status,
        entry.get("reason", ""),
    )
    return entry


async def revoke_permission(app_name: str, resource: str, duration: int) -> None:
    """
    Background task for time-limited (ephemeral) permissions.

    Waits `duration` seconds, then silently flips the permission back OFF.
    """
    await asyncio.sleep(duration)
    app_permissions = PERMISSIONS_STATE.get(app_name)
    if app_permissions and resource in app_permissions:
        app_permissions[resource] = False   # silent revoke


def prune_pending_requests() -> None:
    """Drop abandoned interception records so the queue cannot leak."""
    cutoff = time.monotonic() - PENDING_MAX_AGE_SECONDS
    for request_id, entry in list(pending_requests.items()):
        if entry.get("_created_mono", 0.0) < cutoff:
            pending_requests.pop(request_id, None)


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------
def _redact_secrets(text: object) -> str:
    """Provider errors can echo the request URL; never print the key."""
    message = str(text)
    key = (API_KEY or "").strip()
    if key:
        message = message.replace(key, "[redacted]")
    return re.sub(r"(?i)([?&](?:key|token|api_key)=)[^&\s]+", r"\1[redacted]", message)


def _http_post_json(url: str, payload: dict, headers: dict) -> dict:
    """Minimal stdlib JSON POST (no third-party dependency required)."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=LLM_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def _call_gemini(api_key: str, system_prompt: str, user_prompt: str) -> str:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{LLM_MODEL}:generateContent?key={api_key}"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        },
    }
    data = _http_post_json(url, payload, {"Content-Type": "application/json"})
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _call_openai(api_key: str, system_prompt: str, user_prompt: str) -> str:
    payload = {
        "model": LLM_MODEL,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = _http_post_json(
        "https://api.openai.com/v1/chat/completions", payload, headers
    )
    return data["choices"][0]["message"]["content"]


def _call_provider(api_key: str, system_prompt: str, user_prompt: str) -> str:
    if LLM_PROVIDER == "openai":
        return _call_openai(api_key, system_prompt, user_prompt)
    return _call_gemini(api_key, system_prompt, user_prompt)


async def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Ask the provider, but never let the provider break a request.

    Every failure path is logged explicitly for the developer and returns the
    fail-closed JSON payload (``is_justified: false``) instead of raising or
    returning None. The circuit breaker still short-circuits a known-dead
    provider so a bad key can never add latency.
    """
    api_key = usable_api_key()
    if api_key is None:
        print(
            "[E.D.I.T.H.] LLM error: no usable API key configured "
            "— returning fail-closed verdict."
        )
        return LLM_ERROR_JSON

    if llm_breaker_open():
        remaining = max(0.0, _llm_open_until - time.monotonic())
        print(
            f"[E.D.I.T.H.] LLM error: circuit breaker open for another "
            f"{remaining:.0f}s — returning fail-closed verdict."
        )
        return LLM_ERROR_JSON

    try:
        raw = await asyncio.to_thread(
            _call_provider, api_key, system_prompt, user_prompt
        )
    except urllib.error.HTTPError as exc:
        _record_llm_failure()
        if exc.code in (401, 403):
            print(
                f"[E.D.I.T.H.] LLM error: API key rejected (HTTP {exc.code}) "
                "— check API_KEY / .env — returning fail-closed verdict."
            )
        elif exc.code == 429:
            print(
                "[E.D.I.T.H.] LLM error: rate limited (HTTP 429) "
                "— returning fail-closed verdict."
            )
        elif exc.code == 404:
            print(
                "[E.D.I.T.H.] LLM error: model or endpoint not found (HTTP 404) "
                f"— check LLM_MODEL='{LLM_MODEL}' / LLM_PROVIDER='{LLM_PROVIDER}' "
                "— returning fail-closed verdict."
            )
        else:
            print(
                f"[E.D.I.T.H.] LLM error: provider HTTP {exc.code} "
                f"({_redact_secrets(exc.reason)}) — returning fail-closed verdict."
            )
        return LLM_ERROR_JSON
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        _record_llm_failure()
        print(
            f"[E.D.I.T.H.] LLM error: network/timeout "
            f"({_redact_secrets(exc)}) — returning fail-closed verdict."
        )
        return LLM_ERROR_JSON
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        # Provider replied 200 but the body shape was unexpected.
        _record_llm_failure()
        print(
            f"[E.D.I.T.H.] LLM error: malformed provider response "
            f"({_redact_secrets(exc)}) — returning fail-closed verdict."
        )
        return LLM_ERROR_JSON
    except Exception as exc:  # last-resort guard: a request must never crash
        _record_llm_failure()
        print(
            f"[E.D.I.T.H.] LLM error: unexpected failure "
            f"({_redact_secrets(exc)}) — returning fail-closed verdict."
        )
        return LLM_ERROR_JSON

    if not str(raw or "").strip():
        _record_llm_failure()
        print(
            "[E.D.I.T.H.] LLM error: empty completion "
            "— returning fail-closed verdict."
        )
        return LLM_ERROR_JSON

    _record_llm_success()
    return str(raw)


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def strip_markdown(raw: str) -> str:
    """
    Turn a model reply into bare JSON text.

    Handles ```json ... ``` fences, bare ``` fences, leading "JSON:" labels and
    trailing commentary by keeping only the outermost object braces.
    """
    if not raw:
        return ""

    text = str(raw).strip()

    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start : end + 1]

    return text.strip()


def extract_json(raw: str) -> Optional[dict]:
    """Parse a dict out of an LLM reply, markdown and prose included."""
    cleaned = strip_markdown(raw)
    if not cleaned:
        return None

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Some models emit a trailing comma before the closing brace.
        try:
            parsed = json.loads(re.sub(r",\s*([}\]])", r"\1", cleaned))
        except json.JSONDecodeError:
            return None

    return parsed if isinstance(parsed, dict) else None


async def call_llm_json(system_prompt: str, user_prompt: str) -> Optional[dict]:
    return extract_json(await call_llm(system_prompt, user_prompt))


def coerce_bool(value) -> bool:
    """Models occasionally return "false" / 0 as strings — bool() would lie."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {
            "true", "yes", "1", "justified", "allow", "allowed",
        }
    return False


def fail_closed_verdict(source: str = "llm-error") -> dict:
    """The only answer the firewall trusts when the AI cannot be verified."""
    return {
        "is_justified": False,
        "reason": LLM_ERROR_REASON,
        "source": source,
    }


async def analyse_request(app_name: str, resource: str) -> dict:
    """
    Real-time AI justification — fired when a requested permission is OFF.

    Returns {"is_justified": bool, "reason": str, "source": str}.
    A provider error or unparseable completion yields the fail-closed verdict,
    never a permissive guess.
    """
    system_prompt = STRICT_AUDIT_PROMPT.format(app_name=app_name, resource=resource)
    user_prompt = (
        f"Assess only {app_name}'s current core functionality. "
        f"Resource under review: {resource}. "
        "Reply with the JSON object only — no markdown, no commentary."
    )

    data = await call_llm_json(system_prompt, user_prompt)

    if data is None:
        print(
            "[E.D.I.T.H.] LLM error: completion could not be parsed as JSON "
            "— returning fail-closed verdict."
        )
        return fail_closed_verdict()

    if data.get("reason") == LLM_ERROR_REASON:
        return fail_closed_verdict()

    if "is_justified" not in data:
        print(
            "[E.D.I.T.H.] LLM error: completion is missing 'is_justified' "
            "— returning fail-closed verdict."
        )
        return fail_closed_verdict()

    return {
        "is_justified": coerce_bool(data.get("is_justified")),
        "reason": str(data.get("reason", "")).strip()
        or "No explanation provided.",
        "source": f"llm:{LLM_PROVIDER}",
    }


def fallback_audit() -> dict:
    """Score the matrix locally: penalise every implausible permission left ON."""
    if not PERMISSIONS_STATE:
        return {
            "overall_score": 100,
            "report": (
                "Local policy review: no applications are registered yet, so "
                "there is nothing to audit. Connect an app to begin monitoring."
            ),
            "source": "heuristic",
        }

    offenders = []
    for app_name, permissions in PERMISSIONS_STATE.items():
        for resource, state in permissions.items():
            if state and resource not in PLAUSIBLE_RESOURCES.get(app_name, set()):
                offenders.append(f"{app_name}/{resource}")

    score = max(5, 100 - 12 * len(offenders))
    if offenders:
        report = (
            "Local policy review (offline mode). Unnecessary permissions "
            "are currently ON: " + ", ".join(offenders) + ". "
            "Turn these off unless the app can justify them."
        )
    else:
        report = (
            "Local policy review (offline mode). Every enabled permission "
            "matches the app's core functionality. No action required."
        )
    return {"overall_score": score, "report": report, "source": "heuristic"}


async def warm_up_llm() -> None:
    """
    One quiet probe at startup.

    If the key is missing/expired the breaker trips before the first jury
    interaction, so the demo never waits on a dead provider.
    """
    if usable_api_key() is None:
        return
    try:
        # Two probes: a bad/expired key trips the breaker immediately, before
        # the first jury interaction, so the demo always feels instant.
        for _ in range(LLM_FAILURE_THRESHOLD):
            reply = await call_llm(
                "Reply with JSON only.",
                'Return exactly {"ok": true}.',
            )
            if not is_llm_error_response(reply):
                print("[E.D.I.T.H.] LLM warm-up OK.")
                return
            if llm_breaker_open():
                return
    except Exception as exc:  # never let a warm-up kill the server
        print(f"[E.D.I.T.H.] LLM warm-up skipped ({_redact_secrets(exc)}).")


@asynccontextmanager
async def lifespan(_: FastAPI):
    probe = asyncio.create_task(warm_up_llm())
    try:
        yield
    finally:
        probe.cancel()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="E.D.I.T.H. Core API",
    description="Continuous privacy and permission management system.",
    version="4.1.0",
    lifespan=lifespan,
)

# Local demo: dashboard and app pages are opened from the filesystem (Origin
# "null") or served from any localhost port, so a permissive wildcard is
# correct here. Credentials stay disabled, which is what makes "*" legal.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=600,
)

BASE_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# UI routes (serve the dashboard and demo pages from the same origin)
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_homepage():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/{page_name}.html", response_class=HTMLResponse)
async def serve_html_pages(page_name: str):
    # Resolve inside the project root: a backslash survives the single-segment
    # path converter on Windows, so "../..\secret" would otherwise escape it.
    file_path = (BASE_DIR / f"{page_name}.html").resolve()
    if file_path.parent == BASE_DIR and file_path.is_file():
        return FileResponse(file_path)
    # Returning a dict here would be rendered through HTMLResponse and crash;
    # send an explicit 404 JSON payload instead.
    return JSONResponse({"error": "Page not found"}, status_code=404)


# ---------------------------------------------------------------------------
# Normalisation helpers (case-insensitive lookups)
# ---------------------------------------------------------------------------
def resolve_app_name(app_name: str) -> Optional[str]:
    """Map any casing/whitespace variant onto the canonical app name."""
    if not app_name:
        return None
    return APP_NAME_INDEX.get(app_name.strip().lower())


def resolve_resource(resource: str) -> str:
    """Normalise a resource identifier for dictionary lookups."""
    return (resource or "").strip().lower()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/request_access")
async def request_access(payload: AccessRequest):
    """
    Permission gateway.

    * Permission ON  -> instantly Allowed (logged, velocity-checked).
    * Permission OFF -> the LLM justifies the request, then it is parked in
      pending_requests until an admin decides (real-time interception).
    """
    # --- Normalise immediately to kill any case/whitespace mismatch --------
    app_name = resolve_app_name(payload.app_name)
    resource = resolve_resource(payload.requested_resource)

    if app_name is None or get_trust_score(app_name) is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown application: {payload.app_name}",
        )

    if resource not in KNOWN_RESOURCES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown resource: {payload.requested_resource}",
        )

    prune_pending_requests()
    violations = apply_velocity_tracking(app_name)
    permission_on = PERMISSIONS_STATE.get(app_name, {}).get(resource, False)

    # --- Permission already ON: no interception needed ---------------------
    if permission_on:
        policy_reason = f"'{resource}' is already approved for {app_name}."
        write_log(app_name, resource, "Allowed", policy_reason)
        return {
            "request_id": None,
            "app_name": app_name,
            "requested_resource": resource,
            "status": "Allowed",
            "trust_score": get_trust_score(app_name),
            "is_justified": True,
            "reason": policy_reason,
            "violations": violations,
            "resolved_by": "policy",
        }

    # --- Permission OFF: real-time AI justification, then intercept --------
    analysis = await analyse_request(app_name, resource)

    request_id = uuid.uuid4().hex
    entry = {
        "request_id": request_id,
        "app_name": app_name,
        "resource": resource,
        "status": "pending",
        "current_score": get_trust_score(app_name),
        "is_justified": analysis["is_justified"],
        "reason": analysis["reason"],
        "analysis_source": analysis["source"],
        "violations": violations,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "resolved_by": None,
        "_created_mono": time.monotonic(),
    }
    pending_requests[request_id] = entry

    deadline = time.monotonic() + INTERCEPT_TIMEOUT_SECONDS
    while entry["status"] == "pending" and time.monotonic() < deadline:
        await asyncio.sleep(INTERCEPT_POLL_SECONDS)

    if entry["status"] == "pending":
        # Nobody answered: grant this single request without persisting the
        # permission — the next request will be intercepted again.
        finalize_request(entry, INTERCEPT_TIMEOUT_VERDICT, "auto-timeout")

    result = {
        "request_id": request_id,
        "app_name": app_name,
        "requested_resource": resource,
        "status": entry["status"],
        "trust_score": get_trust_score(app_name),
        "is_justified": analysis["is_justified"],
        "reason": analysis["reason"],
        "violations": violations,
        "resolved_by": entry["resolved_by"],
    }

    pending_requests.pop(request_id, None)
    return result


@app.get("/get_pending")
def get_pending():
    """Return every request still waiting for a human verdict."""
    now = time.monotonic()
    waiting = []
    for entry in pending_requests.values():
        if entry["status"] != "pending":
            continue
        public = {k: v for k, v in entry.items() if not k.startswith("_")}
        elapsed = now - entry.get("_created_mono", now)
        public["expires_in"] = max(
            0, int(round(INTERCEPT_TIMEOUT_SECONDS - elapsed))
        )
        waiting.append(public)

    waiting.sort(key=lambda e: e["created_at"], reverse=True)
    return waiting


@app.post("/resolve_request")
async def resolve_request(payload: ResolveRequest):
    """
    Apply an admin verdict.

    * action = "Blocked" -> deduct 7 from the app's trust score.
    * action = "Allowed" -> flip the permission ON. If a positive `duration`
      is supplied, schedule a background task that silently revokes the
      permission once the time limit expires (ephemeral access).
    """
    action = (payload.action or "").strip().capitalize()
    if action not in ("Allowed", "Blocked"):
        raise HTTPException(
            status_code=400,
            detail="action must be 'Allowed' or 'Blocked'",
        )

    entry = pending_requests.get(payload.request_id)
    if entry is None or entry["status"] != "pending":
        raise HTTPException(
            status_code=404,
            detail=f"No pending request with id: {payload.request_id}",
        )

    finalize_request(entry, action, "admin")
    # The waiting /request_access coroutine still holds a local reference to
    # `entry`, so removing it here is safe and keeps the queue clean.
    pending_requests.pop(payload.request_id, None)

    duration = 0
    expires_at = None

    if action == "Allowed":
        # Grant the permission.
        PERMISSIONS_STATE[entry["app_name"]][entry["resource"]] = True

        # Time-limited access -> schedule the silent auto-revoke.
        if payload.duration and payload.duration > 0:
            duration = payload.duration
            expires_at = (
                datetime.now(timezone.utc)
                + timedelta(seconds=duration)
            ).isoformat()
            asyncio.create_task(
                revoke_permission(
                    entry["app_name"], entry["resource"], duration
                )
            )

    return {
        "request_id": entry["request_id"],
        "app_name": entry["app_name"],
        "resource": entry["resource"],
        "action": action,
        "duration": duration,
        "expires_at": expires_at,
        "trust_score": get_trust_score(entry["app_name"]),
        "message": (
            f"Request resolved as {action}."
            if duration <= 0
            else f"Request resolved as {action} for {duration}s."
        ),
    }


@app.get("/get_notifications")
def get_notifications():
    """Return the notification hub, newest first."""
    return list(reversed(notifications))


@app.get("/run_audit")
async def run_audit():
    """Send the whole ON/OFF permission matrix to the LLM for a security review."""
    # Empty matrix: nothing to send, and no reason to bother the provider.
    if not PERMISSIONS_STATE:
        result = fallback_audit()
        result["analysed_at"] = datetime.now(timezone.utc).isoformat()
        return result

    prompt = (
        "Here is the current ON/OFF state of device permissions: "
        f"{json.dumps(PERMISSIONS_STATE)}. "
        "Act as a cybersecurity auditor. Which ON permissions are risky? "
        'Return JSON: {"overall_score": 1-100, "report": "Summary of risks and '
        'recommendations to turn off"}.'
    )

    data = await call_llm_json(
        "You are a cybersecurity auditor reviewing device permission "
        "assignments. Reply with JSON only.",
        prompt,
    )

    if data is not None and "overall_score" in data:
        try:
            score = int(float(data["overall_score"]))
        except (TypeError, ValueError):
            score = 50

        report = data.get("report", "")
        if isinstance(report, (dict, list)):
            report = json.dumps(report)
        report = str(report).strip() or "No report provided."

        return {
            "overall_score": max(1, min(100, score)),
            "report": report,
            "source": f"llm:{LLM_PROVIDER}",
            "analysed_at": datetime.now(timezone.utc).isoformat(),
        }

    result = fallback_audit()
    result["analysed_at"] = datetime.now(timezone.utc).isoformat()
    return result


@app.get("/get_logs")
def get_logs():
    """Return the 50 most recent access logs (with the AI's reason), newest first."""
    with DB_LOCK:
        cursor = DB_CONNECTION.cursor()
        cursor.execute(
            """
            SELECT id, timestamp, app_name, requested_resource, status, ai_reason
            FROM AccessLogs
            ORDER BY id DESC
            LIMIT 50
            """
        )
        return [dict(row) for row in cursor.fetchall()]


@app.get("/get_trust_history")
def get_trust_history():
    """Return the per-app trust score history for the dashboard line graph."""
    return trust_history


@app.get("/get_scores")
def get_scores():
    """Return the current trust scores of all applications."""
    with DB_LOCK:
        cursor = DB_CONNECTION.cursor()
        cursor.execute(
            "SELECT app_name, trust_score FROM Applications ORDER BY app_name ASC"
        )
        return [dict(row) for row in cursor.fetchall()]


@app.post("/update_score")
def update_score(payload: ScoreUpdate):
    """Manually update an application's trust score (e.g. attack detected)."""
    app_name = resolve_app_name(payload.app_name)
    if app_name is None or get_trust_score(app_name) is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown application: {payload.app_name}",
        )

    set_trust_score(app_name, payload.trust_score)

    return {
        "app_name": app_name,
        "trust_score": payload.trust_score,
        "message": "Trust score updated successfully.",
    }


@app.get("/get_permissions")
def get_permissions():
    """Return the full dictionary of current permission states."""
    return PERMISSIONS_STATE


@app.post("/toggle_permission")
def toggle_permission(payload: PermissionToggle):
    """Flip a single app/resource permission state on or off."""
    app_name = resolve_app_name(payload.app_name)
    if app_name is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown application: {payload.app_name}",
        )

    app_permissions = PERMISSIONS_STATE[app_name]
    resource = resolve_resource(payload.resource)

    if resource not in app_permissions:
        raise HTTPException(
            status_code=404,
            detail=f"Resource '{payload.resource}' is not tracked for {app_name}",
        )

    app_permissions[resource] = payload.state

    return {
        "app_name": app_name,
        "resource": resource,
        "state": payload.state,
        "message": "Permission updated successfully.",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
