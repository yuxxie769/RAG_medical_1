from __future__ import annotations

import base64
import hashlib
import hmac
import os
from datetime import datetime, timezone

from pymongo import ASCENDING, MongoClient, ReturnDocument
from pymongo.errors import PyMongoError


PASSWORD_SCHEME = "pbkdf2_sha256"
PASSWORD_ITERATIONS = 390000


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class AdminAuthUnavailable(RuntimeError):
    """Raised when admin authentication state cannot be read or written safely."""


def normalize_username(username: str) -> str:
    return username.strip().lower()


def hash_password(password: str, *, salt: bytes | None = None, iterations: int = PASSWORD_ITERATIONS) -> str:
    resolved_salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), resolved_salt, iterations)
    return (
        f"{PASSWORD_SCHEME}${iterations}"
        f"${base64.urlsafe_b64encode(resolved_salt).decode('ascii')}"
        f"${base64.urlsafe_b64encode(digest).decode('ascii')}"
    )


def verify_password(password: str, password_digest: str) -> bool:
    try:
        scheme, iteration_text, salt_text, digest_text = password_digest.split("$", 3)
    except ValueError:
        return False
    if scheme != PASSWORD_SCHEME:
        return False
    try:
        iterations = int(iteration_text)
        salt = base64.urlsafe_b64decode(salt_text.encode("ascii"))
        expected = base64.urlsafe_b64decode(digest_text.encode("ascii"))
    except (ValueError, TypeError):
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(actual, expected)


class AdminRepository:
    def __init__(
        self,
        uri: str,
        database: str,
        connect_timeout_ms: int = 3000,
        client: MongoClient | None = None,
    ) -> None:
        self.client = client or MongoClient(
            uri,
            serverSelectionTimeoutMS=connect_timeout_ms,
            connectTimeoutMS=connect_timeout_ms,
        )
        self.database = self.client[database]
        self.users = self.database["admin_users"]
        self._indexes_ready = False

    def ping(self) -> bool:
        try:
            self.client.admin.command("ping")
            self._ensure_indexes()
            return True
        except PyMongoError as exc:
            raise AdminAuthUnavailable(f"MongoDB unavailable: {exc}") from exc

    def _ensure_indexes(self) -> None:
        if self._indexes_ready:
            return
        self.users.create_index([("username_normalized", ASCENDING)], unique=True)
        self._indexes_ready = True

    def ensure_bootstrap_admin(self, username: str, password: str) -> dict | None:
        username = username.strip()
        if not username or not password:
            return None

        self.ping()
        now = _utcnow()
        username_normalized = normalize_username(username)
        existing = self.users.find_one({"username_normalized": username_normalized})
        if existing is not None and not existing.get("seeded_from_env", False):
            return existing

        return self.users.find_one_and_update(
            {"username_normalized": username_normalized},
            {
                "$set": {
                    "username": username,
                    "username_normalized": username_normalized,
                    "password_digest": hash_password(password),
                    "is_active": True,
                    "seeded_from_env": True,
                    "updated_at": now,
                },
                "$setOnInsert": {
                    "created_at": now,
                },
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

    def authenticate(self, username: str, password: str) -> dict | None:
        self.ping()
        user = self.users.find_one({"username_normalized": normalize_username(username)})
        if user is None or not user.get("is_active", False):
            return None
        if not verify_password(password, str(user.get("password_digest") or "")):
            return None
        return user

    def get_active_user(self, username_normalized: str) -> dict | None:
        self.ping()
        return self.users.find_one(
            {
                "username_normalized": username_normalized,
                "is_active": True,
            }
        )
