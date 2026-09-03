"""audit/oracles/wolfram.py -- third-party symbolic/numeric cross-check via
the Wolfram|Alpha Full Results API (plan Task A4).

Independence is the point: this module calls Wolfram|Alpha's own servers to
evaluate an expression and reports back whatever they compute. It never
falls back to a local computation dressed up as third-party confirmation --
if the AppID cannot be read, the network is unreachable, or the API call
fails or returns something unusable for any reason, `wolfram_query` returns
`"UNAVAILABLE: <precise reason>"` and nothing else.

Credential handling (CLAUDE.md "Credentials -- STRICT RULES" /
plan Global Constraints): the Wolfram Alpha AppID lives in
`SCIN Ecosystem/00_System/Credentials/secrets.local.md` (git-ignored,
outside this repo). It is read IN-MEMORY, at call time, by `_read_wolfram_appid`
below, and is never written to a file, printed, logged, or embedded in a
commit message by this module. Callers of this module must uphold the same
rule: never print, log, or persist the return value of `_read_wolfram_appid`.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Absolute path to the vault's secrets file -- a location, not a secret.
SECRETS_PATH = Path(
    r"C:\Users\whtet\Documents\SCIN_ecosystem\SCIN Ecosystem"
    r"\00_System\Credentials\secrets.local.md"
)

_ENTRY_HEADING_RE = re.compile(r"^###[ \t]+(.*Wolfram Alpha.*)$",
                               re.IGNORECASE | re.MULTILINE)
_SECRET_LINE_RE = re.compile(
    r"^-\s*Secret \(key/password/token\):\s*(\S+)\s*$", re.MULTILINE)


def _read_wolfram_appid() -> str | None:
    """Read the Wolfram Alpha AppID in-memory from secrets.local.md.

    Never raises: returns None whenever the file is missing, the "Wolfram
    Alpha" entry heading isn't found, or that entry has no matching
    "Secret (key/password/token):" line -- callers turn None into an
    UNAVAILABLE reason, never a silent fallback. The returned value must
    never be printed, logged, or written anywhere by any caller.
    """
    try:
        text = SECRETS_PATH.read_text(encoding="utf-8")
    except OSError:
        return None

    heading_match = _ENTRY_HEADING_RE.search(text)
    if heading_match is None:
        return None

    # Scope the search to THIS entry only: from the end of its "### " heading
    # to the next "### " heading (or end of file), so a same-named field in a
    # different service's entry can never be matched instead.
    start = heading_match.end()
    rest = text[start:]
    next_heading = re.search(r"^###[ \t]", rest, re.MULTILINE)
    section = rest[:next_heading.start()] if next_heading else rest

    secret_match = _SECRET_LINE_RE.search(section)
    if secret_match is None:
        return None
    return secret_match.group(1)


def wolfram_query(expr: str, timeout: float = 15.0) -> str:
    """Evaluate `expr` via the Wolfram|Alpha Full Results API v2 and return
    its plaintext result pod as a string.

    Returns "UNAVAILABLE: <reason>" -- and ONLY that, never a locally
    computed substitute -- if:
      - the AppID cannot be read from secrets.local.md,
      - the HTTP request fails (network error, timeout, non-200),
      - the response is not valid JSON,
      - Wolfram|Alpha reports the query unsuccessful (queryresult.success
        is false -- e.g. an unparseable or ambiguous `expr`), or
      - no pod in the response carries usable plaintext.
    """
    appid = _read_wolfram_appid()
    if not appid:
        return "UNAVAILABLE: Wolfram Alpha AppID not found in secrets.local.md"

    params = urllib.parse.urlencode({"appid": appid, "input": expr, "output": "JSON"})
    url = f"https://api.wolframalpha.com/v2/query?{params}"

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        return f"UNAVAILABLE: Wolfram Alpha HTTP error {exc.code} ({exc.reason})"
    except urllib.error.URLError as exc:
        return f"UNAVAILABLE: network error contacting Wolfram Alpha ({exc.reason})"
    except OSError as exc:
        return f"UNAVAILABLE: I/O error contacting Wolfram Alpha ({exc})"

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return "UNAVAILABLE: Wolfram Alpha response was not valid JSON"

    query_result = data.get("queryresult", {})
    if not query_result.get("success"):
        error = query_result.get("error")
        if error:
            return f"UNAVAILABLE: Wolfram Alpha reported an error ({error})"
        return ("UNAVAILABLE: Wolfram Alpha did not return a successful result "
                "(query too ambiguous or unparseable)")

    pods = query_result.get("pods", [])
    # Prefer the pod literally titled/id'd "Result".
    for pod in pods:
        if pod.get("id") == "Result" or pod.get("title") == "Result":
            subpods = pod.get("subpods", [])
            if subpods and subpods[0].get("plaintext"):
                return subpods[0]["plaintext"]
    # Fall back to the first non-"Input" pod with plaintext.
    for pod in pods:
        if pod.get("id") == "Input" or pod.get("title") == "Input":
            continue
        subpods = pod.get("subpods", [])
        if subpods and subpods[0].get("plaintext"):
            return subpods[0]["plaintext"]

    return "UNAVAILABLE: Wolfram Alpha returned no pod with usable plaintext"
