from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
from pymongo.errors import PyMongoError


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class ChatUnavailable(RuntimeError):
    """Raised when chat state cannot be read or written safely."""


class ChatRepository:
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
        self.profiles = self.database["chat_profiles"]
        self.conversations = self.database["chat_conversations"]
        self.messages = self.database["chat_messages"]
        self._indexes_ready = False

    def ping(self) -> bool:
        try:
            self.client.admin.command("ping")
            self._ensure_indexes()
            return True
        except PyMongoError as exc:
            raise ChatUnavailable(f"MongoDB unavailable: {exc}") from exc

    def _ensure_indexes(self) -> None:
        if self._indexes_ready:
            return
        self.profiles.create_index([("user_token", ASCENDING)], unique=True)
        self.conversations.create_index([("conversation_id", ASCENDING)], unique=True)
        self.conversations.create_index([("user_token", ASCENDING), ("last_message_at", DESCENDING)])
        self.messages.create_index([("message_id", ASCENDING)], unique=True)
        self.messages.create_index([("conversation_id", ASCENDING), ("created_at", ASCENDING)])
        self._indexes_ready = True

    def get_profile(self, user_token: str | None) -> dict | None:
        if not user_token:
            return None
        self.ping()
        return self.profiles.find_one({"user_token": user_token})

    def upsert_profile(self, user_token: str, nickname: str) -> dict:
        self.ping()
        now = _utcnow()
        return self.profiles.find_one_and_update(
            {"user_token": user_token},
            {
                "$set": {"nickname": nickname, "updated_at": now},
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

    def list_conversations(self, user_token: str) -> list[dict]:
        self.ping()
        return list(
            self.conversations.find({"user_token": user_token}).sort(
                [("last_message_at", DESCENDING), ("created_at", DESCENDING)]
            )
        )

    def get_conversation(self, conversation_id: str, user_token: str) -> dict | None:
        self.ping()
        return self.conversations.find_one({"conversation_id": conversation_id, "user_token": user_token})

    def create_conversation(self, user_token: str, title: str = "新对话") -> dict:
        self.ping()
        now = _utcnow()
        conversation = {
            "conversation_id": uuid4().hex,
            "user_token": user_token,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "last_message_at": now,
        }
        self.conversations.insert_one(conversation)
        return conversation

    def touch_conversation(self, conversation_id: str, user_token: str) -> None:
        now = _utcnow()
        self.conversations.update_one(
            {"conversation_id": conversation_id, "user_token": user_token},
            {"$set": {"updated_at": now, "last_message_at": now}},
        )

    def update_conversation_title(self, conversation_id: str, user_token: str, title: str) -> None:
        now = _utcnow()
        self.conversations.update_one(
            {"conversation_id": conversation_id, "user_token": user_token},
            {
                "$set": {
                    "title": title,
                    "updated_at": now,
                    "last_message_at": now,
                }
            },
        )

    def list_messages(self, conversation_id: str) -> list[dict]:
        self.ping()
        return list(self.messages.find({"conversation_id": conversation_id}).sort([("created_at", ASCENDING)]))

    def create_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        *,
        fallback: bool = False,
        citations: list[dict] | None = None,
    ) -> dict:
        self.ping()
        now = _utcnow()
        message = {
            "message_id": uuid4().hex,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "fallback": fallback,
            "citations": citations or [],
            "created_at": now,
        }
        self.messages.insert_one(message)
        self.conversations.update_one(
            {"conversation_id": conversation_id},
            {"$set": {"updated_at": now, "last_message_at": now}},
        )
        return message

    def get_message(self, conversation_id: str, message_id: str) -> dict | None:
        self.ping()
        return self.messages.find_one({"conversation_id": conversation_id, "message_id": message_id})
