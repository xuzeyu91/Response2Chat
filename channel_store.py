from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional


# The direction is expressed from the upstream protocol to the protocol exposed
# by this proxy.  Keep the original Response -> Chat behavior as the default so
# existing databases, channels, and clients remain compatible after upgrade.
CHANNEL_TYPE_RESPONSE_TO_CHAT = "response_to_chat"
CHANNEL_TYPE_CHAT_TO_RESPONSE = "chat_to_response"
DEFAULT_CHANNEL_TYPE = CHANNEL_TYPE_RESPONSE_TO_CHAT
VALID_CHANNEL_TYPES = {
    CHANNEL_TYPE_RESPONSE_TO_CHAT,
    CHANNEL_TYPE_CHAT_TO_RESPONSE,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_base_url(base_url: str) -> str:
    value = (base_url or "").strip().rstrip("/")
    if not value:
        raise ValueError("渠道 URL 不能为空")

    for suffix in ("/responses", "/chat/completions", "/models"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break

    return value.rstrip("/")


def normalize_channel_type(channel_type: Optional[str]) -> str:
    value = (channel_type or DEFAULT_CHANNEL_TYPE).strip().lower()
    if value not in VALID_CHANNEL_TYPES:
        raise ValueError("渠道转换类型无效")
    return value


def generate_access_key() -> str:
    return f"r2c_{secrets.token_urlsafe(24)}"


def hash_password(password: str, salt_hex: Optional[str] = None) -> str:
    if not password:
        raise ValueError("密码不能为空")

    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        390000,
    )
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt_hex, expected_digest = stored_hash.split("$", 1)
    except ValueError:
        return False

    candidate_digest = hash_password(password, salt_hex).split("$", 1)[1]
    return hmac.compare_digest(candidate_digest, expected_digest)


def mask_secret(secret: str, keep: int = 4) -> str:
    if not secret:
        return "未设置"
    if len(secret) <= keep * 2:
        return "*" * len(secret)
    return f"{secret[:keep]}{'*' * max(len(secret) - keep * 2, 8)}{secret[-keep:]}"


class _ManagedConnection:
    def __init__(self, connection: sqlite3.Connection):
        self._connection = connection

    def __enter__(self) -> sqlite3.Connection:
        return self._connection

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        try:
            if exc_type is None:
                self._connection.commit()
            else:
                self._connection.rollback()
        finally:
            self._connection.close()
        return False


class SettingsStore:
    def __init__(
        self,
        database_path: str,
        default_admin_username: str,
        default_admin_password: str,
    ):
        self.database_path = Path(database_path)
        self.default_admin_username = default_admin_username.strip() or "admin"
        self.default_admin_password = default_admin_password or "admin123456"
        self._channel_cache: Dict[str, Dict[str, Any]] = {}
        self._channel_cache_initialized = False
        self._channel_cache_lock = Lock()

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS admin_users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS channels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT NOT NULL DEFAULT '',
                    upstream_base_url TEXT NOT NULL,
                    upstream_api_key TEXT NOT NULL DEFAULT '',
                    protocol_type TEXT NOT NULL DEFAULT 'response_to_chat',
                    access_key TEXT NOT NULL UNIQUE,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            # SQLite does not apply new CREATE TABLE columns to an existing
            # table.  Make the schema migration explicit and idempotent.
            channel_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(channels)").fetchall()
            }
            if "protocol_type" not in channel_columns:
                conn.execute(
                    "ALTER TABLE channels "
                    "ADD COLUMN protocol_type TEXT NOT NULL "
                    "DEFAULT 'response_to_chat'"
                )
            conn.commit()

        self._ensure_default_admin()
        self._reload_channel_cache()

    def _connect(self) -> _ManagedConnection:
        conn = sqlite3.connect(str(self.database_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return _ManagedConnection(conn)

    def _ensure_default_admin(self) -> None:
        with self._connect() as conn:
            existing = conn.execute("SELECT COUNT(1) AS total FROM admin_users").fetchone()
            if existing and existing["total"] > 0:
                return

            now = utc_now_iso()
            conn.execute(
                """
                INSERT INTO admin_users (username, password_hash, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    self.default_admin_username,
                    hash_password(self.default_admin_password),
                    now,
                    now,
                ),
            )
            conn.commit()

    def authenticate_admin(self, username: str, password: str) -> bool:
        if not username or not password:
            return False

        with self._connect() as conn:
            row = conn.execute(
                "SELECT password_hash FROM admin_users WHERE username = ? LIMIT 1",
                (username.strip(),),
            ).fetchone()
            if not row:
                return False
            return verify_password(password, row["password_hash"])

        return False

    def change_admin_password(self, username: str, current_password: str, new_password: str) -> tuple[bool, str]:
        if not new_password:
            return False, "新密码不能为空"
        if len(new_password) < 8:
            return False, "新密码至少需要 8 位"
        if not self.authenticate_admin(username, current_password):
            return False, "当前密码不正确"

        with self._connect() as conn:
            now = utc_now_iso()
            conn.execute(
                "UPDATE admin_users SET password_hash = ?, updated_at = ? WHERE username = ?",
                (hash_password(new_password), now, username.strip()),
            )
            conn.commit()

        return True, "管理员密码已更新"

    def list_channels(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, name, description, upstream_base_url, upstream_api_key, protocol_type,
                       access_key, enabled, created_at, updated_at
                FROM channels
                ORDER BY updated_at DESC, id DESC
                """
            ).fetchall()
        return [self._row_to_channel(row) for row in rows]

    def count_channels(self) -> Dict[str, int]:
        with self._connect() as conn:
            total_row = conn.execute(
                "SELECT COUNT(1) AS total, SUM(enabled) AS enabled_total FROM channels"
            ).fetchone()

        total = int(total_row["total"] or 0) if total_row else 0
        enabled_total = int(total_row["enabled_total"] or 0) if total_row else 0
        return {
            "total": total,
            "enabled": enabled_total,
            "disabled": max(total - enabled_total, 0),
        }

    def get_channel(self, channel_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, name, description, upstream_base_url, upstream_api_key, protocol_type,
                       access_key, enabled, created_at, updated_at
                FROM channels
                WHERE id = ?
                LIMIT 1
                """,
                (channel_id,),
            ).fetchone()

        return self._row_to_channel(row) if row else None

    def get_channel_by_access_key(self, access_key: str) -> Optional[Dict[str, Any]]:
        if not access_key:
            return None

        with self._channel_cache_lock:
            cache_initialized = self._channel_cache_initialized
            channel = self._channel_cache.get(access_key)

        if not cache_initialized:
            self._reload_channel_cache()
            with self._channel_cache_lock:
                channel = self._channel_cache.get(access_key)

        return dict(channel) if channel else None

    def create_channel(
        self,
        name: str,
        base_url: str,
        upstream_api_key: str,
        description: str = "",
        protocol_type: str = DEFAULT_CHANNEL_TYPE,
    ) -> Dict[str, Any]:
        clean_name = (name or "").strip()
        if not clean_name:
            raise ValueError("渠道名称不能为空")

        normalized_url = normalize_base_url(base_url)
        normalized_protocol_type = normalize_channel_type(protocol_type)
        now = utc_now_iso()

        with self._connect() as conn:
            access_key = self._generate_unique_access_key(conn)
            try:
                cursor = conn.execute(
                    """
                    INSERT INTO channels (
                        name, description, upstream_base_url, upstream_api_key,
                        protocol_type, access_key, enabled, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
                    """,
                    (
                        clean_name,
                        (description or "").strip(),
                        normalized_url,
                        (upstream_api_key or "").strip(),
                        normalized_protocol_type,
                        access_key,
                        now,
                        now,
                    ),
                )
                conn.commit()
            except sqlite3.IntegrityError as exc:
                raise ValueError("渠道名称已存在，请更换后重试") from exc

        if cursor.lastrowid is None:
            raise RuntimeError("渠道创建失败，未返回有效的记录 ID")

        channel = self.get_channel(int(cursor.lastrowid))
        if not channel:
            raise RuntimeError("渠道创建失败，保存后未能读取配置")

        self._cache_channel(channel)
        return channel

    def update_channel(
        self,
        channel_id: int,
        name: str,
        base_url: str,
        upstream_api_key: Optional[str],
        description: str,
        enabled: bool,
        clear_upstream_api_key: bool = False,
        protocol_type: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        existing = self.get_channel(channel_id)
        if not existing:
            return None

        clean_name = (name or "").strip()
        if not clean_name:
            raise ValueError("渠道名称不能为空")

        normalized_url = normalize_base_url(base_url)
        normalized_protocol_type = normalize_channel_type(
            protocol_type if protocol_type is not None else existing["protocol_type"]
        )
        next_upstream_api_key = existing["upstream_api_key"]
        if clear_upstream_api_key:
            next_upstream_api_key = ""
        elif upstream_api_key is not None and upstream_api_key.strip():
            next_upstream_api_key = upstream_api_key.strip()

        with self._connect() as conn:
            try:
                conn.execute(
                    """
                    UPDATE channels
                    SET name = ?,
                        description = ?,
                        upstream_base_url = ?,
                        upstream_api_key = ?,
                        protocol_type = ?,
                        enabled = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        clean_name,
                        (description or "").strip(),
                        normalized_url,
                        next_upstream_api_key,
                        normalized_protocol_type,
                        1 if enabled else 0,
                        utc_now_iso(),
                        channel_id,
                    ),
                )
                conn.commit()
            except sqlite3.IntegrityError as exc:
                raise ValueError("渠道名称已存在，请更换后重试") from exc

        channel = self.get_channel(channel_id)
        if channel:
            self._cache_channel(channel)
        return channel

    def set_channel_enabled(self, channel_id: int, enabled: bool) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            conn.execute(
                "UPDATE channels SET enabled = ?, updated_at = ? WHERE id = ?",
                (1 if enabled else 0, utc_now_iso(), channel_id),
            )
            conn.commit()

        channel = self.get_channel(channel_id)
        if channel:
            self._cache_channel(channel)
        return channel

    def rotate_access_key(self, channel_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM channels WHERE id = ?", (channel_id,)).fetchone()
            if not row:
                return None

            conn.execute(
                "UPDATE channels SET access_key = ?, updated_at = ? WHERE id = ?",
                (self._generate_unique_access_key(conn), utc_now_iso(), channel_id),
            )
            conn.commit()

        channel = self.get_channel(channel_id)
        if channel:
            self._cache_channel(channel)
        return channel

    def delete_channel(self, channel_id: int) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM channels WHERE id = ?", (channel_id,))
            conn.commit()
            deleted = cursor.rowcount > 0

        if deleted:
            self._remove_channel_from_cache(channel_id)
        return deleted

    def _reload_channel_cache(self) -> None:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, name, description, upstream_base_url, upstream_api_key, protocol_type,
                       access_key, enabled, created_at, updated_at
                FROM channels
                """
            ).fetchall()

        channels = [self._row_to_channel(row) for row in rows]
        with self._channel_cache_lock:
            self._channel_cache = {
                channel["access_key"]: channel
                for channel in channels
            }
            self._channel_cache_initialized = True

    def _cache_channel(self, channel: Dict[str, Any]) -> None:
        cached_channel = dict(channel)
        with self._channel_cache_lock:
            stale_keys = [
                access_key
                for access_key, cached in self._channel_cache.items()
                if cached["id"] == cached_channel["id"]
            ]
            for access_key in stale_keys:
                self._channel_cache.pop(access_key, None)
            self._channel_cache[cached_channel["access_key"]] = cached_channel
            self._channel_cache_initialized = True

    def _remove_channel_from_cache(self, channel_id: int) -> None:
        with self._channel_cache_lock:
            stale_keys = [
                access_key
                for access_key, cached in self._channel_cache.items()
                if cached["id"] == channel_id
            ]
            for access_key in stale_keys:
                self._channel_cache.pop(access_key, None)

    def _generate_unique_access_key(self, conn: sqlite3.Connection) -> str:
        while True:
            access_key = generate_access_key()
            exists = conn.execute(
                "SELECT 1 FROM channels WHERE access_key = ? LIMIT 1",
                (access_key,),
            ).fetchone()
            if not exists:
                return access_key

    def _row_to_channel(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "name": row["name"],
            "description": row["description"],
            "upstream_base_url": row["upstream_base_url"],
            "upstream_api_key": row["upstream_api_key"],
            "protocol_type": normalize_channel_type(row["protocol_type"]),
            "access_key": row["access_key"],
            "enabled": bool(row["enabled"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }


class AdminSessionManager:
    def __init__(self, ttl_seconds: int = 12 * 60 * 60):
        self.ttl_seconds = max(ttl_seconds, 300)
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()

    def create_session(self, username: str) -> str:
        token = secrets.token_urlsafe(32)
        expires_at = time.time() + self.ttl_seconds
        with self._lock:
            self._cleanup_locked()
            self._sessions[token] = {
                "username": username,
                "expires_at": expires_at,
            }
        return token

    def get_username(self, token: Optional[str]) -> Optional[str]:
        if not token:
            return None

        now = time.time()
        with self._lock:
            session = self._sessions.get(token)
            if not session:
                return None
            if session["expires_at"] <= now:
                self._sessions.pop(token, None)
                return None

            session["expires_at"] = now + self.ttl_seconds
            return str(session["username"])

    def revoke(self, token: Optional[str]) -> None:
        if not token:
            return

        with self._lock:
            self._sessions.pop(token, None)

    def _cleanup_locked(self) -> None:
        now = time.time()
        expired_tokens = [token for token, session in self._sessions.items() if session["expires_at"] <= now]
        for token in expired_tokens:
            self._sessions.pop(token, None)
