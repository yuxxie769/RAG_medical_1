from __future__ import annotations

from datetime import datetime

from fastapi.testclient import TestClient

from app.api import main as api_main
from app.api.main import app
from app.chat import ChatUnavailable


def _message(role: str, content: str, *, fallback: bool = False, citations: list[dict] | None = None) -> dict:
    return {
        "message_id": f"msg_{role}_{abs(hash(content))}",
        "conversation_id": "conv_1",
        "role": role,
        "content": content,
        "fallback": fallback,
        "citations": citations or [],
        "created_at": datetime(2026, 6, 3, 12, 0, 0),
    }


class FakeChatService:
    def __init__(self) -> None:
        self.profile = None
        self.conversations = []
        self.messages: dict[str, list[dict]] = {}

    def build_page_state(
        self,
        user_token,
        *,
        conversation_id=None,
        top_k=5,
        profile_error=None,
        question_error=None,
        question_value="",
        citation_error=None,
    ):
        active = None
        if self.profile and self.conversations:
            active = next((item for item in self.conversations if item["conversation_id"] == conversation_id), None)
            active = active or self.conversations[0]
        return {
            "service_available": True,
            "profile": self.profile,
            "conversations": self.conversations,
            "active_conversation": active,
            "messages": self.messages.get(active["conversation_id"], []) if active else [],
            "top_k": top_k,
            "profile_error": profile_error,
            "question_error": question_error,
            "question_value": question_value,
            "citation_error": citation_error,
            "active_citation": None,
        }

    def save_profile(self, user_token, nickname):
        self.profile = {"user_token": user_token, "nickname": nickname}
        if not self.conversations:
            self.conversations = [
                {
                    "conversation_id": "conv_1",
                    "title": "新对话",
                    "last_message_at": datetime(2026, 6, 3, 12, 0, 0),
                }
            ]
        return self.profile

    def create_conversation(self, user_token):
        conversation = {
            "conversation_id": f"conv_{len(self.conversations) + 1}",
            "title": "新对话",
            "last_message_at": datetime(2026, 6, 3, 12, 0, 0),
        }
        self.conversations.insert(0, conversation)
        self.messages[conversation["conversation_id"]] = []
        return conversation

    def ask_question(self, user_token, conversation_id, query, top_k):
        if not query.strip():
            raise ValueError("请输入问题")
        user_message = _message("user", query)
        if query == "fallback":
            assistant_message = _message("assistant", "当前暂时无法完成回答：测试错误", fallback=True)
        else:
            assistant_message = _message(
                "assistant",
                "这里是生成回答[1]",
                citations=[
                    {
                        "doc_id": "doc_1",
                        "content": "Q: 口干怎么办\nA: 建议补水",
                        "score": 0.92,
                        "metadata": {"source_path": "tests/sample_data.jsonl", "source_line_no": 3},
                    }
                ],
            )
        self.messages.setdefault(conversation_id, []).extend([user_message, assistant_message])

    def get_citation(self, user_token, conversation_id, message_id, citation_index):
        messages = self.messages.get(conversation_id, [])
        for message in messages:
            if message["message_id"] == message_id:
                return message["citations"][citation_index]
        raise KeyError("未找到引用来源")


def test_ui_home_renders_profile_modal(monkeypatch):
    monkeypatch.setitem(api_main._DATASTORE, "chat_service", FakeChatService())

    response = TestClient(app).get("/ui")

    assert response.status_code == 200
    assert "先给自己起个昵称" in response.text


def test_profile_submission_sets_cookie(monkeypatch):
    monkeypatch.setitem(api_main._DATASTORE, "chat_service", FakeChatService())

    response = TestClient(app).post("/ui/profile", data={"nickname": "小林"})

    assert response.status_code == 200
    assert "set-cookie" in response.headers
    assert "小林" in response.text
    assert "新对话" in response.text


def test_create_conversation_renders_new_conversation(monkeypatch):
    service = FakeChatService()
    service.save_profile("token_1", "小林")
    monkeypatch.setitem(api_main._DATASTORE, "chat_service", service)

    client = TestClient(app)
    client.cookies.set("rag_medical_user_token", "token_1")
    response = client.post("/ui/conversations")

    assert response.status_code == 200
    assert response.text.count("新对话") >= 1


def test_show_conversation_loads_history(monkeypatch):
    service = FakeChatService()
    service.save_profile("token_1", "小林")
    service.messages["conv_1"] = [_message("user", "口干怎么办"), _message("assistant", "这里是生成回答")]
    monkeypatch.setitem(api_main._DATASTORE, "chat_service", service)

    client = TestClient(app)
    client.cookies.set("rag_medical_user_token", "token_1")
    response = client.get("/ui/conversations/conv_1")

    assert response.status_code == 200
    assert "口干怎么办" in response.text
    assert "这里是生成回答" in response.text


def test_answer_body_renders_clickable_inline_citation(monkeypatch):
    service = FakeChatService()
    service.save_profile("token_1", "小林")
    service.messages["conv_1"] = [
        _message(
            "assistant",
            "分析如下[1]",
            citations=[
                {
                    "doc_id": "doc_inline_1",
                    "content": "inline citation content",
                    "score": 0.88,
                    "metadata": {"source_path": "tests/sample_data.jsonl", "source_line_no": 8},
                }
            ],
        )
    ]
    monkeypatch.setitem(api_main._DATASTORE, "chat_service", service)

    client = TestClient(app)
    client.cookies.set("rag_medical_user_token", "token_1")
    response = client.get("/ui/conversations/conv_1")

    assert response.status_code == 200
    assert 'class="inline-citation"' in response.text
    assert "/ui/conversations/conv_1/messages/" in response.text
    assert "/citations/0" in response.text


def test_ask_question_renders_fallback_message(monkeypatch):
    service = FakeChatService()
    service.save_profile("token_1", "小林")
    monkeypatch.setitem(api_main._DATASTORE, "chat_service", service)

    client = TestClient(app)
    client.cookies.set("rag_medical_user_token", "token_1")
    response = client.post("/ui/conversations/conv_1/ask", data={"query": "fallback", "top_k": "5"})

    assert response.status_code == 200
    assert "谨慎回答" in response.text
    assert "当前暂时无法完成回答：测试错误" in response.text


def test_ask_question_validation_error_is_rendered(monkeypatch):
    service = FakeChatService()
    service.save_profile("token_1", "小林")
    monkeypatch.setitem(api_main._DATASTORE, "chat_service", service)

    client = TestClient(app)
    client.cookies.set("rag_medical_user_token", "token_1")
    response = client.post("/ui/conversations/conv_1/ask", data={"query": "", "top_k": "5"})

    assert response.status_code == 422
    assert "请输入问题" in response.text


def test_citation_detail_renders_content(monkeypatch):
    service = FakeChatService()
    service.save_profile("token_1", "小林")
    assistant = _message(
        "assistant",
        "这里是生成回答",
        citations=[
            {
                "doc_id": "doc_1",
                "content": "Q: 口干怎么办\nA: 建议补水",
                "score": 0.92,
                "metadata": {"source_path": "tests/sample_data.jsonl", "source_line_no": 3},
            }
        ],
    )
    service.messages["conv_1"] = [assistant]
    monkeypatch.setitem(api_main._DATASTORE, "chat_service", service)

    client = TestClient(app)
    client.cookies.set("rag_medical_user_token", "token_1")
    response = client.get(
        f"/ui/conversations/conv_1/messages/{assistant['message_id']}/citations/0"
    )

    assert response.status_code == 200
    assert "建议补水" in response.text
    assert "tests/sample_data.jsonl" not in response.text
    assert "0.920" not in response.text


def test_citation_detail_handles_out_of_range(monkeypatch):
    service = FakeChatService()
    service.save_profile("token_1", "小林")
    assistant = _message("assistant", "这里是生成回答", citations=[])
    service.messages["conv_1"] = [assistant]
    monkeypatch.setitem(api_main._DATASTORE, "chat_service", service)

    client = TestClient(app)
    client.cookies.set("rag_medical_user_token", "token_1")
    response = client.get(f"/ui/conversations/conv_1/messages/{assistant['message_id']}/citations/9")

    assert response.status_code == 404


def test_ui_home_handles_chat_unavailable(monkeypatch):
    class BrokenChatService:
        def build_page_state(self, *args, **kwargs):
            raise ChatUnavailable("MongoDB unavailable")

    monkeypatch.setitem(api_main._DATASTORE, "chat_service", BrokenChatService())

    response = TestClient(app).get("/ui")

    assert response.status_code == 503
    assert "当前无法进入问答功能" in response.text
