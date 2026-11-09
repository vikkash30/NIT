# E.D.I.T.H. — Zero-Trust Permission Gateway

**NITS Hacks 9.0 prototype.** E.D.I.T.H. sits between applications and sensitive
device resources (contacts, camera, microphone, location) and decides every
request at runtime. Permissions that are switched **on** are allowed instantly;
permissions switched **off** are escalated to an LLM that must justify the
request against the app's *current* core functionality. Anything the AI cannot
justify — or any AI/provider failure — is **blocked (fail closed)**.

## Architecture

```
┌──────────────┐   fetch    ┌───────────────────────────┐
│  Demo apps   │ ─────────► │  FastAPI gateway main.py  │
│  paytm.html  │            │  127.0.0.1:8000           │
│  ypay.html   │ ◄───────── │                           │
│  weather.html│   verdict  │  1. permission lookup     │
│ calculator   │            │  2. LLM justification     │
└──────────────┘            │     (strict JSON verdict) │
                            │  3. pending request queue │
┌──────────────┐   poll     │  4. SQLite audit log      │
│  index.html  │ ◄────────► │     (in-memory)           │
│  dashboard   │  resolve   └───────────────────────────┘
└──────────────┘                        │
                                        ▼
                             Google Gemini (or OpenAI)
                             via stdlib urllib, 15 s cap
```

- **Gateway** — `main.py` (FastAPI). One async endpoint, `/request_access`,
  returns `Allowed`/`Blocked`; denied-but-unjustified requests are parked in
  `pending_requests` and surfaced in the dashboard until an admin resolves them
  or the 30 s hold expires.
- **AI layer** — `call_llm()` posts the strict prompt
  *“You are an enterprise Zero-Trust Firewall…”* and parses a JSON verdict.
  Markdown fences are stripped before `json.loads()`. Invalid keys, 401/403,
  429, timeouts, malformed/empty replies and missing fields all return
  `{"is_justified": false, "reason": "API Error: Defaulting to Zero-Trust Block"}`.
  A circuit breaker (2 failures → 180 s cooldown) and a startup warm-up probe
  keep a dead key from hanging a live demo.
- **State** — SQLite (`:memory:`) holds permissions, scores, logs, notification
  history and trust history for the duration of the process.
- **Dashboard** — `index.html` renders posture score, live audit, pending
  intercepts and the permission matrix.

| Route | Method | Purpose |
| --- | --- | --- |
| `/request_access` | POST | Gate an app's resource request |
| `/get_pending` | GET | Pending intercepts for the dashboard |
| `/resolve_request` | POST | Admin allow/deny (one-shot) |
| `/get_notifications` | GET | Toast feed |
| `/run_audit` | GET | Full local + AI policy audit |
| `/get_logs` | GET | Request history |
| `/get_trust_history` | GET | Trust score timeline |
| `/get_scores` | GET | Current trust score |
| `/update_score` | POST | Adjust trust score |
| `/get_permissions` | GET | Permission matrix |
| `/toggle_permission` | POST | Flip a permission on/off |

## Run it

```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt

# .env next to main.py — never commit this file
echo GEMINI_API_KEY_1=your_key_here > .env

uvicorn main:app --host 127.0.0.1 --port 8000
```

Open the dashboard at `http://127.0.0.1:8000/` (or serve the folder and open
`index.html`), then demo an app page such as `paytm.html`. Interactive API
docs: `http://127.0.0.1:8000/docs`.

Optional environment variables: `LLM_PROVIDER` (`gemini`/`openai`), `LLM_MODEL`
(default `gemini-3.5-flash-lite`), `GEMINI_API_KEY_2`/`_3` for failover.

## Tests

```bash
python test_main.py          # 23 offline checks, stubbed provider
python test_main.py --live   # one real provider round-trip
python test_main.py --models # list models available to the configured key
```

## License

MIT — see [LICENSE](LICENSE).
