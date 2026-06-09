from __future__ import annotations

from fastapi import HTTPException

from app.chat.repository import ChatRepository


DEFAULT_TOP_K = 5
DEFAULT_CONVERSATION_TITLE = "新对话"


class ChatService:
    def __init__(self, repository: ChatRepository) -> None:
        self.repository = repository

    def ping(self) -> bool:
        return self.repository.ping()

    def get_profile(self, user_token: str | None) -> dict | None:
        return self.repository.get_profile(user_token)

    def save_profile(self, user_token: str, nickname: str) -> dict:
        nickname = nickname.strip()
        if not nickname:
            raise ValueError("昵称不能为空")
        profile = self.repository.upsert_profile(user_token, nickname)
        if not self.repository.list_conversations(user_token):
            self.repository.create_conversation(user_token, DEFAULT_CONVERSATION_TITLE)
        return profile

    def create_conversation(self, user_token: str) -> dict:
        self._require_profile(user_token)
        return self.repository.create_conversation(user_token, DEFAULT_CONVERSATION_TITLE)

    def build_page_state(
        self,
        user_token: str | None,
        *,
        conversation_id: str | None = None,
        top_k: int = DEFAULT_TOP_K,
        profile_error: str | None = None,
        question_error: str | None = None,
        question_value: str = "",
        citation_error: str | None = None,
    ) -> dict:
        profile = self.repository.get_profile(user_token)
        if profile is None:
            return {
                "service_available": True,
                "profile": None,
                "conversations": [],
                "active_conversation": None,
                "messages": [],
                "top_k": top_k,
                "profile_error": profile_error,
                "question_error": question_error,
                "question_value": question_value,
                "citation_error": citation_error,
                "active_citation": None,
            }

        conversations = self.repository.list_conversations(profile["user_token"])
        if not conversations:
            active_conversation = self.repository.create_conversation(profile["user_token"], DEFAULT_CONVERSATION_TITLE)
            conversations = [active_conversation]
        else:
            active_conversation = None
            if conversation_id:
                active_conversation = next(
                    (item for item in conversations if item["conversation_id"] == conversation_id),
                    None,
                )
            if active_conversation is None:
                active_conversation = conversations[0]

        messages = self.repository.list_messages(active_conversation["conversation_id"])
        return {
            "service_available": True,
            "profile": profile,
            "conversations": conversations,
            "active_conversation": active_conversation,
            "messages": messages,
            "top_k": top_k,
            "profile_error": profile_error,
            "question_error": question_error,
            "question_value": question_value,
            "citation_error": citation_error,
            "active_citation": None,
        }

    def ask_question(self, user_token: str, conversation_id: str, query: str, top_k: int) -> None:
        profile = self._require_profile(user_token)
        conversation = self._require_conversation(conversation_id, profile["user_token"])
        query = query.strip()
        if not query:
            raise ValueError("请输入问题")

        self.repository.create_message(conversation["conversation_id"], "user", query)
        if conversation.get("title") == DEFAULT_CONVERSATION_TITLE:
            self.repository.update_conversation_title(
                conversation["conversation_id"],
                profile["user_token"],
                self._title_from_query(query),
            )

        answer_text, fallback, citations, retrieval_results = self._generate_answer(query, top_k)
        self.repository.create_message(
            conversation["conversation_id"],
            "assistant",
            answer_text,
            fallback=fallback,
            citations=citations,
            retrieval_results=retrieval_results,
        )

    def get_citation(self, user_token: str, conversation_id: str, message_id: str, citation_index: int) -> dict:
        profile = self._require_profile(user_token)
        self._require_conversation(conversation_id, profile["user_token"])
        message = self.repository.get_message(conversation_id, message_id)
        if message is None or message.get("role") != "assistant":
            raise KeyError("未找到引用来源")
        citations = message.get("citations") or []
        if citation_index < 0 or citation_index >= len(citations):
            raise IndexError("引用索引超出范围")
        return citations[citation_index]

    def get_retrieval_result(self, user_token: str, conversation_id: str, message_id: str, result_index: int) -> dict:
        profile = self._require_profile(user_token)
        self._require_conversation(conversation_id, profile["user_token"])
        message = self.repository.get_message(conversation_id, message_id)
        if message is None or message.get("role") != "assistant":
            raise KeyError("未找到检索结果")
        retrieval_results = message.get("retrieval_results") or []
        if result_index < 0 or result_index >= len(retrieval_results):
            raise IndexError("检索结果索引超出范围")
        return retrieval_results[result_index]

    def _generate_answer(self, query: str, top_k: int) -> tuple[str, bool, list[dict], list[dict]]:
        from app.api import main as api_main

        try:
            response = api_main.answer_query(api_main.AnswerRequest(query=query, top_k=top_k))
            citations = [item.model_dump(mode="json") for item in response.citations]
            retrieval_results = [item.model_dump(mode="json") for item in getattr(response, "retrieval_results", [])]
            return response.answer, response.fallback, citations, retrieval_results
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else "问答服务暂时不可用"
            return f"当前暂时无法完成回答：{detail}", True, [], []

    def _require_profile(self, user_token: str | None) -> dict:
        profile = self.repository.get_profile(user_token)
        if profile is None:
            raise PermissionError("当前会话未设置昵称")
        return profile

    def _require_conversation(self, conversation_id: str, user_token: str) -> dict:
        conversation = self.repository.get_conversation(conversation_id, user_token)
        if conversation is None:
            raise KeyError("未找到对应会话")
        return conversation

    def _title_from_query(self, query: str) -> str:
        compact = " ".join(query.split())
        return compact[:18] + ("..." if len(compact) > 18 else "")
