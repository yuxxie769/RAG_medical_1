from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import datetime, timezone


ADMIN_COOKIE_NAME = "rag_medical_admin_session"


class AdminSessionError(ValueError):
    """Raised when an admin session cookie is missing, invalid, or cannot be trusted."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _sign(secret: str, payload: str) -> str:
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def build_admin_session_value(secret: str, username_normalized: str) -> str:
    if not secret.strip():
        raise AdminSessionError("Admin session secret is not configured.")
    payload = json.dumps(
        {
            "username_normalized": username_normalized,
            "issued_at": _utcnow().isoformat(),
        },
        separators=(",", ":"),
        ensure_ascii=True,
    )
    payload_b64 = base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii")
    return f"{payload_b64}.{_sign(secret, payload_b64)}"


def read_admin_session_value(secret: str, cookie_value: str | None) -> str:
    if not secret.strip():
        raise AdminSessionError("Admin session secret is not configured.")
    if not cookie_value:
        raise AdminSessionError("Admin authentication required.")
    try:
        payload_b64, signature = cookie_value.split(".", 1)
    except ValueError as exc:
        raise AdminSessionError("Admin authentication required.") from exc
    expected_signature = _sign(secret, payload_b64)
    if not hmac.compare_digest(signature, expected_signature):
        raise AdminSessionError("Admin authentication required.")
    try:
        payload = json.loads(base64.urlsafe_b64decode(payload_b64.encode("ascii")).decode("utf-8"))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise AdminSessionError("Admin authentication required.") from exc
    username_normalized = str(payload.get("username_normalized") or "").strip().lower()
    if not username_normalized:
        raise AdminSessionError("Admin authentication required.")
    return username_normalized
