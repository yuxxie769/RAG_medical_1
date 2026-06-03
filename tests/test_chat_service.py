from __future__ import annotations

from fastapi import HTTPException

from app.chat.service import ChatService


class FakeRepository:
    def __init__(self):
        self.profile = {"user_token": "token_1", "nickname": "小林"}
        self.conversation = {"conversation_id": "conv_1", "user_token": "token_1", "title": "新对话"}
        self.messages = []

    def ping(self):
        return True

    def get_profile(self, user_token):
        return self.profile if user_token == "token_1" else None

    def list_conversations(self, user_token):
        return [self.conversation]

    def get_conversation(self, conversation_id, user_token):
        if conversation_id == "conv_1" and user_token == "token_1":
            return self.conversation
        return None

    def create_message(self, conversation_id, role, content, *, fallback=False, citations=None):
        self.messages.append(
            {
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "fallback": fallback,
                "citations": citations or [],
            }
        )
        return self.messages[-1]

    def update_conversation_title(self, conversation_id, user_token, title):
        self.conversation["title"] = title

    def get_message(self, conversation_id, message_id):
        return None


def test_chat_service_turns_answer_http_error_into_fallback(monkeypatch):
    repository = FakeRepository()
    service = ChatService(repository=repository)

    def broken_answer_query(_request):
        raise HTTPException(status_code=503, detail="No documents indexed yet. Call /ingest first.")

    monkeypatch.setattr("app.api.main.answer_query", broken_answer_query)

    service.ask_question("token_1", "conv_1", "口干怎么办", 5)

    assert repository.messages[0]["role"] == "user"
    assert repository.messages[1]["role"] == "assistant"
    assert repository.messages[1]["fallback"] is True
    assert "No documents indexed yet" in repository.messages[1]["content"]
