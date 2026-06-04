from __future__ import annotations

from fastapi.testclient import TestClient

from app.admin import ADMIN_COOKIE_NAME, AdminRepository, build_admin_session_value, hash_password, verify_password
from app.api import main as api_main
from app.api.main import app


class FakeAdminCollection:
    def __init__(self) -> None:
        self.documents: dict[str, dict] = {}
        self.indexes: list[tuple] = []

    def create_index(self, keys, unique=False):
        self.indexes.append((tuple(keys), unique))

    def find_one(self, query):
        username_normalized = query.get("username_normalized")
        if username_normalized is None:
            return None
        document = self.documents.get(username_normalized)
        if document is None:
            return None
        if query.get("is_active") is True and not document.get("is_active"):
            return None
        return dict(document)

    def find_one_and_update(self, query, update, upsert=False, return_document=None):
        username_normalized = query["username_normalized"]
        current = dict(self.documents.get(username_normalized) or {})
        if not current and not upsert:
            return None
        if not current:
            current.update(update.get("$setOnInsert", {}))
        current.update(update.get("$set", {}))
        self.documents[username_normalized] = current
        return dict(current)


class FakeAdminCommand:
    def command(self, name):
        assert name == "ping"
        return {"ok": 1}


class FakeAdminClient:
    def __init__(self, collection: FakeAdminCollection) -> None:
        self.admin = FakeAdminCommand()
        self._database = {"admin_users": collection}

    def __getitem__(self, database_name):
        return self._database


class FakeAdminRepository:
    def __init__(self) -> None:
        self.user = {"username": "admin", "username_normalized": "admin", "is_active": True}

    def authenticate(self, username, password):
        if username == "admin" and password == "secret":
            return self.user
        return None

    def get_active_user(self, username_normalized):
        if username_normalized == "admin":
            return self.user
        return None


class FakeIngestRepository:
    def ping(self):
        return True

    def get_status(self, ingest_run_id):
        return None

    def find_latest_run_for_source_path(self, source_path, collection_name):
        if source_path.endswith("sample_data.jsonl"):
            return {
                "ingest_run_id": "run_lookup",
                "status": "running",
                "stage": "ingesting",
                "source_path": source_path,
                "progress_percent": 48.0,
                "current_scanned_line": 48,
                "total_lines": 100,
                "processed_batches": 2,
                "succeeded_batches": 1,
                "pending_batches": 1,
                "failed_batches": 0,
                "execution_outcome": "processed",
                "lines_per_second": 12.0,
                "batches_per_second": 0.5,
                "eta_seconds": 5.0,
                "cancel_requested": False,
                "failed_batch_details": [],
            }
        return None


def test_password_hash_roundtrip():
    digest = hash_password("secret")

    assert verify_password("secret", digest) is True
    assert verify_password("wrong", digest) is False


def test_admin_repository_bootstrap_creates_and_updates_seeded_user():
    collection = FakeAdminCollection()
    repo = AdminRepository(
        uri="mongodb://unused",
        database="unused",
        client=FakeAdminClient(collection),
    )

    created = repo.ensure_bootstrap_admin("Admin", "secret")
    updated = repo.ensure_bootstrap_admin("Admin", "new-secret")

    assert created["username_normalized"] == "admin"
    assert verify_password("new-secret", updated["password_digest"]) is True
    assert ((("username_normalized", 1),), True) in collection.indexes


def test_admin_login_page_renders(monkeypatch):
    response = TestClient(app).get("/ui/admin/login")

    assert response.status_code == 200
    assert "登录" in response.text


def test_admin_login_sets_cookie_and_redirects(monkeypatch):
    monkeypatch.setitem(api_main._DATASTORE, "admin_repository", FakeAdminRepository())
    monkeypatch.setattr(api_main.settings, "admin_session_secret", "web-admin-secret")

    response = TestClient(app).post(
        "/ui/admin/login",
        data={"username": "admin", "password": "secret", "next": "/ui/admin/ingest"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/ui/admin/ingest"
    assert "set-cookie" in response.headers
    assert ADMIN_COOKIE_NAME in response.headers["set-cookie"]


def test_admin_login_failure_renders_error(monkeypatch):
    monkeypatch.setitem(api_main._DATASTORE, "admin_repository", FakeAdminRepository())

    response = TestClient(app).post("/ui/admin/login", data={"username": "admin", "password": "bad"})

    assert response.status_code == 401
    assert "密码" in response.text


def test_admin_ingest_requires_login(monkeypatch):
    response = TestClient(app).get("/ui/admin/ingest", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"].startswith("/ui/admin/login?next=")


def test_admin_ingest_page_renders_after_login(monkeypatch):
    monkeypatch.setitem(api_main._DATASTORE, "admin_repository", FakeAdminRepository())
    monkeypatch.setitem(api_main._DATASTORE, "ingest_repository", FakeIngestRepository())
    monkeypatch.setattr(api_main.settings, "admin_session_secret", "web-admin-secret")

    client = TestClient(app)
    client.cookies.set(ADMIN_COOKIE_NAME, build_admin_session_value("web-admin-secret", "admin"))
    response = client.get("/ui/admin/ingest")

    assert response.status_code == 200
    assert "导入" in response.text
    assert "任务" in response.text


def test_admin_ingest_lookup_by_source_path_renders_existing_run(monkeypatch):
    monkeypatch.setitem(api_main._DATASTORE, "admin_repository", FakeAdminRepository())
    monkeypatch.setitem(api_main._DATASTORE, "ingest_repository", FakeIngestRepository())
    monkeypatch.setattr(api_main.settings, "admin_session_secret", "web-admin-secret")
    monkeypatch.setattr(api_main.settings, "milvus_collection_name", "test_collection")

    client = TestClient(app)
    client.cookies.set(ADMIN_COOKIE_NAME, build_admin_session_value("web-admin-secret", "admin"))
    response = client.get("/ui/admin/ingest", params={"lookup_source_path": "tests/sample_data.jsonl"})

    assert response.status_code == 200
    assert "run_lookup" in response.text
    assert "sample_data.jsonl" in response.text


def test_admin_ingest_lookup_by_source_path_returns_not_found(monkeypatch):
    monkeypatch.setitem(api_main._DATASTORE, "admin_repository", FakeAdminRepository())
    monkeypatch.setitem(api_main._DATASTORE, "ingest_repository", FakeIngestRepository())
    monkeypatch.setattr(api_main.settings, "admin_session_secret", "web-admin-secret")
    monkeypatch.setattr(api_main.settings, "milvus_collection_name", "test_collection")

    client = TestClient(app)
    client.cookies.set(ADMIN_COOKIE_NAME, build_admin_session_value("web-admin-secret", "admin"))
    response = client.get("/ui/admin/ingest", params={"lookup_source_path": "tests/missing.jsonl"})

    assert response.status_code == 404
    assert "未找到路径对应的导入任务" in response.text
