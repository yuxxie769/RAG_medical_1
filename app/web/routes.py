from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlencode
from uuid import uuid4

try:
    import bleach
except ImportError:  # pragma: no cover - optional dependency
    bleach = None

try:
    import markdown
except ImportError:  # pragma: no cover - optional dependency
    markdown = None
from fastapi import APIRouter, BackgroundTasks, FastAPI, Form, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape

from app.admin import ADMIN_COOKIE_NAME, AdminAuthUnavailable, AdminSessionError, build_admin_session_value
from app.chat import ChatUnavailable


USER_COOKIE_NAME = "rag_medical_user_token"
DEFAULT_TOP_K = 5
TOP_K_OPTIONS = (3, 5, 8)
TERMINAL_INGEST_STATUSES = {"cancelled", "completed", "completed_with_errors", "failed"}
LIVE_INGEST_STATUSES = {"queued", "running", "cancelling"}

router = APIRouter(include_in_schema=False)
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))
CITATION_REF_PATTERN = re.compile(r"\[(\d+)\]")


ALLOWED_TAGS = [
    "p", "br", "hr",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "strong", "b", "em", "i", "u", "s", "del",
    "ul", "ol", "li",
    "code", "pre",
    "blockquote",
    "a", "abbr",
    "table", "thead", "tbody", "tr", "th", "td",
    "sup", "sub",
]
ALLOWED_ATTRS = {"a": ["href", "title"], "abbr": ["title"]}


def _render_message_content(message: dict, active_conversation: dict | None) -> Markup:
    content = str(message.get("content") or "")
    if not content:
        return Markup("")

    html_body = bleached_markdown(content)

    if message.get("role") != "assistant" or not active_conversation:
        return Markup(html_body)

    citations = message.get("citations") or []
    if not citations:
        return Markup(html_body)

    conversation_id = active_conversation.get("conversation_id")
    message_id = message.get("message_id")
    if not conversation_id or not message_id:
        return Markup(html_body)

    def _citation_chip(match):
        citation_number = int(match.group(1))
        citation_index = citation_number - 1
        if 0 <= citation_index < len(citations):
            citation_url = (
                f"/ui/conversations/{conversation_id}"
                f"/messages/{message_id}/citations/{citation_index}"
            )
            return (
                f'<button type="button" class="inline-citation" '
                f'hx-get="{citation_url}" hx-target="#citation-detail" '
                f'hx-swap="innerHTML">[{citation_number}]</button>'
            )
        return match.group(0)

    return Markup(CITATION_REF_PATTERN.sub(_citation_chip, html_body))


def bleached_markdown(text: str) -> str:
    """Convert markdown to sanitized HTML."""
    if markdown is None or bleach is None:
        return str(escape(text)).replace("\n", "<br>")
    html = markdown.markdown(text, extensions=["extra", "sane_lists"])
    return bleach.clean(html, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS)


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


def _normalize_next_path(next_path: str | None, default: str = "/ui/admin/ingest") -> str:
    if not next_path:
        return default
    if not next_path.startswith("/") or next_path.startswith("//"):
        return default
    if next_path.startswith("/ui/admin/login"):
        return default
    return next_path


def _build_login_redirect(next_path: str | None = None) -> str:
    query = urlencode({"next": _normalize_next_path(next_path)})
    return f"/ui/admin/login?{query}"


def _admin_page_redirect(next_path: str | None = None) -> RedirectResponse:
    return RedirectResponse(url=_build_login_redirect(next_path), status_code=status.HTTP_303_SEE_OTHER)


def _admin_current_path(request: Request) -> str:
    return str(request.url.path) + (f"?{request.url.query}" if request.url.query else "")


def _authenticate_admin_request(request: Request) -> dict:
    from app.api import main as api_main

    return api_main._authenticate_admin_cookie(request.cookies.get(ADMIN_COOKIE_NAME))


def _get_ingest_service():
    from app.api import main as api_main

    return api_main._get_ingest_service()


def _get_ingest_repository():
    from app.api import main as api_main

    return api_main._get_ingest_repository()


def _get_admin_repository():
    from app.api import main as api_main

    return api_main._get_admin_repository()


def _get_admin_settings():
    from app.api import main as api_main

    return api_main.settings


def _get_admin_health_state() -> dict:
    try:
        _get_ingest_repository().ping()
        mongodb_status = "ok"
        health_error = None
    except Exception as exc:
        mongodb_status = "unavailable"
        health_error = str(exc)
    return {
        "mongodb_status": mongodb_status,
        "health_error": health_error,
    }


def _default_ingest_form_values() -> dict:
    return {
        "source_path": "",
        "raw_batch_id": "",
        "clean_batch_id": "",
        "kb_batch_id": "",
        "flush_to_milvus": True,
        "force_reingest": False,
    }


def _normalize_source_path_value(source_path: str) -> str:
    candidate = source_path.strip()
    if not candidate:
        return ""
    try:
        return str(Path(candidate).resolve())
    except OSError:
        return str(Path(candidate).absolute())


def _ingest_urls(ingest_run_id: str) -> dict[str, str]:
    return {
        "status_url": f"/ui/admin/ingest/status?ingest_run_id={ingest_run_id}",
        "events_url": f"/ingest/{ingest_run_id}/events",
        "cancel_url": f"/ui/admin/ingest/{ingest_run_id}",
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


def _render_admin_login(
    request: Request,
    *,
    username_value: str = "",
    login_error: str | None = None,
    next_path: str | None = None,
    status_code: int = status.HTTP_200_OK,
    full_page: bool = False,
) -> HTMLResponse:
    context = _base_context(request)
    context.update(
        {
            "username_value": username_value,
            "login_error": login_error,
            "next_path": _normalize_next_path(next_path),
        }
    )
    template_name = "admin_login.html" if full_page else "fragments/admin_login_shell.html"
    return templates.TemplateResponse(request, template_name, context, status_code=status_code)


def _render_ingest_status_fragment(
    request: Request,
    *,
    ingest_state: dict | None = None,
    ingest_status_error: str | None = None,
    status_code: int = status.HTTP_200_OK,
) -> HTMLResponse:
    if ingest_state is not None:
        ingest_state = {**ingest_state, **_ingest_urls(ingest_state["ingest_run_id"])}
    context = _base_context(request)
    context.update(
        {
            "ingest_state": ingest_state,
            "ingest_status_error": ingest_status_error,
            "terminal_ingest_statuses": TERMINAL_INGEST_STATUSES,
            "live_ingest_statuses": LIVE_INGEST_STATUSES,
        }
    )
    return templates.TemplateResponse(
        request,
        "fragments/admin_ingest_status.html",
        context,
        status_code=status_code,
    )


def _render_admin_ingest(
    request: Request,
    *,
    admin_user: dict,
    ingest_form_values: dict | None = None,
    lookup_source_path_value: str = "",
    ingest_form_error: str | None = None,
    ingest_state: dict | None = None,
    ingest_status_error: str | None = None,
    status_code: int = status.HTTP_200_OK,
    full_page: bool = False,
) -> HTMLResponse:
    if ingest_state is not None:
        ingest_state = {**ingest_state, **_ingest_urls(ingest_state["ingest_run_id"])}
    context = _base_context(request)
    context.update(
        {
            "admin_user": admin_user,
            "ingest_form_values": ingest_form_values or _default_ingest_form_values(),
            "lookup_source_path_value": lookup_source_path_value,
            "ingest_form_error": ingest_form_error,
            "ingest_state": ingest_state,
            "ingest_status_error": ingest_status_error,
            "health_state": _get_admin_health_state(),
            "terminal_ingest_statuses": TERMINAL_INGEST_STATUSES,
            "live_ingest_statuses": LIVE_INGEST_STATUSES,
        }
    )
    template_name = "admin_ingest.html" if full_page else "fragments/admin_ingest_shell.html"
    return templates.TemplateResponse(request, template_name, context, status_code=status_code)


@router.get("/ui", response_class=HTMLResponse)
def ui_home(request: Request) -> HTMLResponse:
    return _render_shell(
        request,
        user_token=request.cookies.get(USER_COOKIE_NAME),
        full_page=True,
    )


@router.get("/ui/admin/login", response_class=HTMLResponse)
def admin_login_page(request: Request, next: str | None = None) -> HTMLResponse:
    try:
        _authenticate_admin_request(request)
    except PermissionError:
        return _render_admin_login(
            request,
            next_path=next,
            full_page=True,
        )
    except AdminAuthUnavailable as exc:
        return _render_admin_login(
            request,
            next_path=next,
            login_error=str(exc),
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            full_page=True,
        )
    return RedirectResponse(url=_normalize_next_path(next), status_code=status.HTTP_303_SEE_OTHER)


@router.post("/ui/admin/login", response_class=HTMLResponse)
def admin_login_submit(
    request: Request,
    username: str = Form(default=""),
    password: str = Form(default=""),
    next: str = Form(default="/ui/admin/ingest"),
) -> HTMLResponse:
    resolved_next = _normalize_next_path(next)
    try:
        admin_user = _get_admin_repository().authenticate(username, password)
    except AdminAuthUnavailable as exc:
        return _render_admin_login(
            request,
            username_value=username.strip(),
            login_error=str(exc),
            next_path=resolved_next,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            full_page=not _is_fragment_request(request),
        )

    if admin_user is None:
        return _render_admin_login(
            request,
            username_value=username.strip(),
            login_error="用户名或密码错误",
            next_path=resolved_next,
            status_code=status.HTTP_401_UNAUTHORIZED,
            full_page=not _is_fragment_request(request),
        )

    try:
        cookie_value = build_admin_session_value(
            _get_admin_settings().admin_session_secret,
            admin_user["username_normalized"],
        )
    except AdminSessionError as exc:
        return _render_admin_login(
            request,
            username_value=username.strip(),
            login_error=str(exc),
            next_path=resolved_next,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            full_page=not _is_fragment_request(request),
        )

    response = RedirectResponse(url=resolved_next, status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(
        ADMIN_COOKIE_NAME,
        cookie_value,
        httponly=True,
        samesite="lax",
    )
    return response


@router.post("/ui/admin/logout")
def admin_logout(request: Request) -> RedirectResponse:
    response = RedirectResponse(url="/ui/admin/login", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie(ADMIN_COOKIE_NAME)
    return response


@router.get("/ui/admin/ingest", response_class=HTMLResponse)
def admin_ingest_home(request: Request, lookup_source_path: str | None = None) -> HTMLResponse:
    try:
        admin_user = _authenticate_admin_request(request)
    except PermissionError:
        return _admin_page_redirect(_admin_current_path(request))
    except AdminAuthUnavailable as exc:
        return _render_admin_login(
            request,
            login_error=str(exc),
            next_path=_admin_current_path(request),
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            full_page=True,
        )

    normalized_lookup_source_path = _normalize_source_path_value(lookup_source_path or "")
    if not normalized_lookup_source_path:
        return _render_admin_ingest(request, admin_user=admin_user, full_page=True)

    try:
        ingest_state = _get_ingest_repository().find_latest_run_for_source_path(
            normalized_lookup_source_path,
            _get_admin_settings().milvus_collection_name,
        )
    except Exception as exc:
        from app.ingest import MongoUnavailable

        if isinstance(exc, MongoUnavailable):
            return _render_admin_ingest(
                request,
                admin_user=admin_user,
                lookup_source_path_value=lookup_source_path or "",
                ingest_status_error=str(exc),
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                full_page=True,
            )
        raise

    if ingest_state is None:
        return _render_admin_ingest(
            request,
            admin_user=admin_user,
            lookup_source_path_value=lookup_source_path or "",
            ingest_status_error=f"未找到路径对应的导入任务：{normalized_lookup_source_path}",
            status_code=status.HTTP_404_NOT_FOUND,
            full_page=True,
        )

    return _render_admin_ingest(
        request,
        admin_user=admin_user,
        lookup_source_path_value=lookup_source_path or "",
        ingest_state=ingest_state,
        full_page=True,
    )


@router.post("/ui/admin/ingest", response_class=HTMLResponse)
def admin_ingest_submit(
    request: Request,
    background_tasks: BackgroundTasks,
    source_path: str = Form(default=""),
    raw_batch_id: str = Form(default=""),
    clean_batch_id: str = Form(default=""),
    kb_batch_id: str = Form(default=""),
    flush_to_milvus: str | None = Form(default=None),
    force_reingest: str | None = Form(default=None),
) -> HTMLResponse:
    try:
        admin_user = _authenticate_admin_request(request)
    except PermissionError:
        return _admin_page_redirect(_admin_current_path(request))
    except AdminAuthUnavailable as exc:
        return _render_admin_login(
            request,
            login_error=str(exc),
            next_path=_admin_current_path(request),
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            full_page=not _is_fragment_request(request),
        )

    ingest_form_values = {
        "source_path": source_path.strip(),
        "raw_batch_id": raw_batch_id.strip(),
        "clean_batch_id": clean_batch_id.strip(),
        "kb_batch_id": kb_batch_id.strip(),
        "flush_to_milvus": flush_to_milvus is not None,
        "force_reingest": force_reingest is not None,
    }
    if not ingest_form_values["source_path"]:
        return _render_admin_ingest(
            request,
            admin_user=admin_user,
            ingest_form_values=ingest_form_values,
            ingest_form_error="请输入数据源路径",
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            full_page=not _is_fragment_request(request),
        )

    try:
        prepared_run = _get_ingest_service().prepare_ingest(
            source_path=ingest_form_values["source_path"],
            raw_batch_id=ingest_form_values["raw_batch_id"] or None,
            clean_batch_id=ingest_form_values["clean_batch_id"] or None,
            kb_batch_id=ingest_form_values["kb_batch_id"] or None,
            flush_to_milvus=ingest_form_values["flush_to_milvus"],
            force_reingest=ingest_form_values["force_reingest"],
            target_collection_name=_get_admin_settings().milvus_collection_name,
        )
        background_tasks.add_task(_get_ingest_service().execute_prepared_ingest, prepared_run)
        ingest_state = _get_ingest_repository().get_status(prepared_run.ingest_run_id)
    except FileNotFoundError as exc:
        return _render_admin_ingest(
            request,
            admin_user=admin_user,
            ingest_form_values=ingest_form_values,
            ingest_form_error=str(exc),
            status_code=status.HTTP_400_BAD_REQUEST,
            full_page=not _is_fragment_request(request),
        )
    except Exception as exc:
        from app.ingest import IngestLeaseConflict, MongoUnavailable

        if isinstance(exc, IngestLeaseConflict):
            status_code = status.HTTP_409_CONFLICT
        elif isinstance(exc, MongoUnavailable):
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        else:
            raise
        return _render_admin_ingest(
            request,
            admin_user=admin_user,
            ingest_form_values=ingest_form_values,
            ingest_form_error=str(exc),
            status_code=status_code,
            full_page=not _is_fragment_request(request),
        )

    return _render_admin_ingest(
        request,
        admin_user=admin_user,
        ingest_form_values=ingest_form_values,
        ingest_state=ingest_state,
        full_page=not _is_fragment_request(request),
    )


@router.get("/ui/admin/ingest/status", response_class=HTMLResponse)
def admin_ingest_status(request: Request, ingest_run_id: str) -> HTMLResponse:
    try:
        _authenticate_admin_request(request)
    except PermissionError:
        return _admin_page_redirect(_admin_current_path(request))
    except AdminAuthUnavailable as exc:
        return _render_admin_login(
            request,
            login_error=str(exc),
            next_path=_admin_current_path(request),
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            full_page=not _is_fragment_request(request),
        )
    try:
        ingest_state = _get_ingest_repository().get_status(ingest_run_id)
    except Exception as exc:
        from app.ingest import MongoUnavailable

        if isinstance(exc, MongoUnavailable):
            return _render_ingest_status_fragment(
                request,
                ingest_status_error=str(exc),
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        raise
    if ingest_state is None:
        return _render_ingest_status_fragment(
            request,
            ingest_status_error=f"未找到导入任务：{ingest_run_id}",
            status_code=status.HTTP_404_NOT_FOUND,
        )
    return _render_ingest_status_fragment(request, ingest_state=ingest_state)


@router.delete("/ui/admin/ingest/{ingest_run_id}", response_class=HTMLResponse)
def admin_cancel_ingest(request: Request, ingest_run_id: str) -> HTMLResponse:
    try:
        _authenticate_admin_request(request)
    except PermissionError:
        return _admin_page_redirect(_admin_current_path(request))
    except AdminAuthUnavailable as exc:
        return _render_admin_login(
            request,
            login_error=str(exc),
            next_path=_admin_current_path(request),
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            full_page=not _is_fragment_request(request),
        )
    try:
        ingest_state = _get_ingest_repository().request_cancel(ingest_run_id)
    except Exception as exc:
        from app.ingest import MongoUnavailable

        if isinstance(exc, MongoUnavailable):
            return _render_ingest_status_fragment(
                request,
                ingest_status_error=str(exc),
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        raise
    if ingest_state is None:
        return _render_ingest_status_fragment(
            request,
            ingest_status_error=f"未找到导入任务：{ingest_run_id}",
            status_code=status.HTTP_404_NOT_FOUND,
        )
    return _render_ingest_status_fragment(request, ingest_state=ingest_state)


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
