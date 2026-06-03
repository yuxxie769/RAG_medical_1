from __future__ import annotations

from pathlib import Path
import re
from uuid import uuid4

from fastapi import APIRouter, FastAPI, Form, Request, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape

from app.chat import ChatUnavailable


USER_COOKIE_NAME = "rag_medical_user_token"
DEFAULT_TOP_K = 5
TOP_K_OPTIONS = (3, 5, 8)

router = APIRouter(include_in_schema=False)
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))
CITATION_REF_PATTERN = re.compile(r"\[(\d+)\]")


def _render_message_content(message: dict, active_conversation: dict | None) -> Markup:
    content = str(message.get("content") or "")
    if message.get("role") != "assistant" or not active_conversation:
        return Markup(escape(content))

    citations = message.get("citations") or []
    if not citations:
        return Markup(escape(content))

    conversation_id = active_conversation.get("conversation_id")
    message_id = message.get("message_id")
    if not conversation_id or not message_id:
        return Markup(escape(content))

    rendered_parts: list[str] = []
    last_index = 0
    for match in CITATION_REF_PATTERN.finditer(content):
        start, end = match.span()
        rendered_parts.append(str(escape(content[last_index:start])))

        citation_number = int(match.group(1))
        citation_index = citation_number - 1
        if 0 <= citation_index < len(citations):
            citation_url = (
                f"/ui/conversations/{escape(str(conversation_id))}"
                f"/messages/{escape(str(message_id))}/citations/{citation_index}"
            )
            rendered_parts.append(
                f'<button type="button" class="inline-citation" '
                f'hx-get="{citation_url}" hx-target="#citation-detail" '
                f'hx-swap="innerHTML">[{citation_number}]</button>'
            )
        else:
            rendered_parts.append(str(escape(match.group(0))))
        last_index = end

    rendered_parts.append(str(escape(content[last_index:])))
    return Markup("".join(rendered_parts))


templates.env.globals["render_message_content"] = _render_message_content


def mount_web(app: FastAPI) -> None:
    static_dir = Path(__file__).resolve().parent / "static"
    if not getattr(app.state, "web_static_mounted", False):
        app.mount("/ui/static", StaticFiles(directory=str(static_dir)), name="web-static")
        app.state.web_static_mounted = True
    app.include_router(router)


def _chat_service():
    from app.api import main as api_main

    return api_main._get_chat_service()


def _is_fragment_request(request: Request) -> bool:
    return request.headers.get("HX-Request") == "true"


def _base_context(request: Request) -> dict:
    from app.api import main as api_main

    return {
        "request": request,
        "project_name": api_main.settings.project_name,
        "top_k_options": TOP_K_OPTIONS,
    }


def _render_shell(
    request: Request,
    *,
    user_token: str | None,
    conversation_id: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    profile_error: str | None = None,
    question_error: str | None = None,
    question_value: str = "",
    citation_error: str | None = None,
    full_page: bool = False,
    status_code: int = status.HTTP_200_OK,
) -> HTMLResponse:
    context = _base_context(request)
    context["user_cookie_name"] = USER_COOKIE_NAME
    try:
        state = _chat_service().build_page_state(
            user_token,
            conversation_id=conversation_id,
            top_k=top_k,
            profile_error=profile_error,
            question_error=question_error,
            question_value=question_value,
            citation_error=citation_error,
        )
    except ChatUnavailable as exc:
        state = {
            "service_available": False,
            "service_error": str(exc),
            "profile": None,
            "conversations": [],
            "active_conversation": None,
            "messages": [],
            "top_k": top_k,
            "profile_error": None,
            "question_error": None,
            "question_value": "",
            "citation_error": None,
            "active_citation": None,
        }
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    context.update(state)
    template_name = "chat.html" if full_page else "fragments/chat_shell.html"
    return templates.TemplateResponse(request, template_name, context, status_code=status_code)


@router.get("/ui", response_class=HTMLResponse)
def ui_home(request: Request) -> HTMLResponse:
    return _render_shell(
        request,
        user_token=request.cookies.get(USER_COOKIE_NAME),
        full_page=True,
    )


@router.post("/ui/profile", response_class=HTMLResponse)
def save_profile(request: Request, nickname: str = Form(default="")) -> HTMLResponse:
    user_token = request.cookies.get(USER_COOKIE_NAME) or uuid4().hex
    nickname = nickname.strip()
    if not nickname:
        return _render_shell(
            request,
            user_token=request.cookies.get(USER_COOKIE_NAME),
            profile_error="请输入昵称",
            full_page=not _is_fragment_request(request),
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        )

    try:
        _chat_service().save_profile(user_token, nickname)
    except ChatUnavailable as exc:
        return _render_shell(
            request,
            user_token=user_token,
            profile_error=str(exc),
            full_page=not _is_fragment_request(request),
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    response = _render_shell(request, user_token=user_token, full_page=not _is_fragment_request(request))
    response.set_cookie(
        USER_COOKIE_NAME,
        user_token,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 30,
    )
    return response


@router.post("/ui/conversations", response_class=HTMLResponse)
def create_conversation(request: Request) -> HTMLResponse:
    user_token = request.cookies.get(USER_COOKIE_NAME)
    if not user_token:
        return _render_shell(
            request,
            user_token=None,
            profile_error="请先设置昵称",
            full_page=not _is_fragment_request(request),
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
    try:
        conversation = _chat_service().create_conversation(user_token)
    except PermissionError:
        return _render_shell(
            request,
            user_token=None,
            profile_error="请先设置昵称",
            full_page=not _is_fragment_request(request),
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
    except ChatUnavailable as exc:
        return _render_shell(
            request,
            user_token=user_token,
            question_error=str(exc),
            full_page=not _is_fragment_request(request),
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    return _render_shell(
        request,
        user_token=user_token,
        conversation_id=conversation["conversation_id"],
        full_page=not _is_fragment_request(request),
    )


@router.get("/ui/conversations/{conversation_id}", response_class=HTMLResponse)
def show_conversation(request: Request, conversation_id: str) -> HTMLResponse:
    user_token = request.cookies.get(USER_COOKIE_NAME)
    if not user_token:
        return _render_shell(
            request,
            user_token=None,
            profile_error="请先设置昵称",
            full_page=not _is_fragment_request(request),
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
    return _render_shell(
        request,
        user_token=user_token,
        conversation_id=conversation_id,
        full_page=not _is_fragment_request(request),
    )


@router.post("/ui/conversations/{conversation_id}/ask", response_class=HTMLResponse)
def ask_question(
    request: Request,
    conversation_id: str,
    query: str = Form(default=""),
    top_k: int = Form(default=DEFAULT_TOP_K),
) -> HTMLResponse:
    user_token = request.cookies.get(USER_COOKIE_NAME)
    if not user_token:
        return _render_shell(
            request,
            user_token=None,
            profile_error="请先设置昵称",
            full_page=not _is_fragment_request(request),
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
    try:
        _chat_service().ask_question(user_token, conversation_id, query, top_k)
    except ValueError as exc:
        return _render_shell(
            request,
            user_token=user_token,
            conversation_id=conversation_id,
            top_k=top_k,
            question_error=str(exc),
            question_value=query,
            full_page=not _is_fragment_request(request),
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        )
    except PermissionError:
        return _render_shell(
            request,
            user_token=None,
            profile_error="请先设置昵称",
            full_page=not _is_fragment_request(request),
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
    except KeyError:
        return _render_shell(
            request,
            user_token=user_token,
            question_error="未找到对应会话",
            top_k=top_k,
            full_page=not _is_fragment_request(request),
            status_code=status.HTTP_404_NOT_FOUND,
        )
    except ChatUnavailable as exc:
        return _render_shell(
            request,
            user_token=user_token,
            conversation_id=conversation_id,
            top_k=top_k,
            question_error=str(exc),
            question_value=query,
            full_page=not _is_fragment_request(request),
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    return _render_shell(
        request,
        user_token=user_token,
        conversation_id=conversation_id,
        top_k=top_k,
        full_page=not _is_fragment_request(request),
    )


@router.get(
    "/ui/conversations/{conversation_id}/messages/{message_id}/citations/{citation_index}",
    response_class=HTMLResponse,
)
def show_citation(
    request: Request,
    conversation_id: str,
    message_id: str,
    citation_index: int,
) -> HTMLResponse:
    user_token = request.cookies.get(USER_COOKIE_NAME)
    context = _base_context(request)
    if not user_token:
        return templates.TemplateResponse(
            request,
            "fragments/citation_detail.html",
            {
                **context,
                "citation": None,
                "citation_error": "请先设置昵称",
            },
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
    try:
        citation = _chat_service().get_citation(user_token, conversation_id, message_id, citation_index)
        citation_error = None
        status_code = status.HTTP_200_OK
    except (PermissionError, KeyError, IndexError) as exc:
        citation = None
        citation_error = str(exc)
        status_code = status.HTTP_404_NOT_FOUND
    except ChatUnavailable as exc:
        citation = None
        citation_error = str(exc)
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return templates.TemplateResponse(
        request,
        "fragments/citation_detail.html",
        {
            **context,
            "citation": citation,
            "citation_error": citation_error,
        },
        status_code=status_code,
    )
