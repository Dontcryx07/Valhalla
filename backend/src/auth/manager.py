"""
In-memory session-based authentication for Valhalla web admin.

Admin credentials come from the ADMIN_CREDENTIALS env var as
email:password pairs separated by semicolons. Sessions are stored
in-memory (lost on server restart).
"""

import os
import hashlib
import hmac
import secrets
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

SESSION_EXPIRY_HOURS = 24


class AuthManager:
    def __init__(self):
        self._users: dict[str, dict] = {}
        self._sessions: dict[str, dict] = {}
        self._load_users()

    def _load_users(self):
        from src import config as _cfg

        raw = _cfg.ADMIN_CREDENTIALS
        if not raw:
            logger.warning("[Auth] No ADMIN_CREDENTIALS configured — login disabled")
            return
        for pair in raw.split(";"):
            pair = pair.strip()
            if ":" not in pair:
                continue
            email, password = pair.split(":", 1)
            email = email.strip().lower()
            password = password.strip()
            if email and password:
                self._users[email] = {
                    "name": email.split("@")[0],
                    "password_hash": self._hash_password(password),
                }
                logger.info("[Auth] Loaded admin user: %s", email)

    def _hash_password(self, password: str) -> str:
        salt = os.urandom(32)
        key = hashlib.scrypt(
            password.encode("utf-8"), salt=salt, n=16384, r=8, p=1, dklen=64
        )
        return salt.hex() + ":" + key.hex()

    def verify_password(self, password: str, stored_hash: str) -> bool:
        try:
            salt_hex, key_hex = stored_hash.split(":")
            salt = bytes.fromhex(salt_hex)
            stored_key = bytes.fromhex(key_hex)
            computed = hashlib.scrypt(
                password.encode("utf-8"), salt=salt, n=16384, r=8, p=1, dklen=64
            )
            return hmac.compare_digest(computed, stored_key)
        except (ValueError, AttributeError):
            return False

    def authenticate(self, email: str, password: str) -> Optional[dict]:
        """Verify credentials and return user dict, or None."""
        user = self._users.get(email.lower().strip())
        if not user:
            return None
        if not self.verify_password(password, user["password_hash"]):
            return None
        return {"email": email.lower().strip(), "name": user["name"], "role": "admin"}

    def create_session(self, email: str) -> str:
        user = self._users.get(email.lower())
        if not user:
            return ""
        token = secrets.token_urlsafe(32)
        self._sessions[token] = {
            "email": email.lower(),
            "name": user["name"],
            "role": "admin",
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (
                datetime.utcnow() + timedelta(hours=SESSION_EXPIRY_HOURS)
            ).isoformat(),
        }
        return token

    def validate_session(self, token: str) -> Optional[dict]:
        if not token:
            return None
        session = self._sessions.get(token)
        if not session:
            return None
        try:
            expires = datetime.fromisoformat(session["expires_at"])
        except (ValueError, TypeError):
            return None
        if datetime.utcnow() > expires:
            self._sessions.pop(token, None)
            return None
        return {
            "email": session["email"],
            "name": session["name"],
            "role": session["role"],
        }

    def revoke_session(self, token: str):
        self._sessions.pop(token, None)


auth_manager = AuthManager()
