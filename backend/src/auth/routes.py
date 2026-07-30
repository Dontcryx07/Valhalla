"""
Auth API routes — login, logout, session check.
"""

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import Optional

from src.auth.manager import auth_manager

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginInput(BaseModel):
    email: str
    password: str


class LogoutInput(BaseModel):
    token: str


@router.post("/login")
async def login(request: LoginInput):
    """Verify credentials and return a session token."""
    user = auth_manager.authenticate(request.email, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = auth_manager.create_session(request.email)
    return {"token": token, "user": user}


@router.post("/logout")
async def logout(request: LogoutInput):
    """Revoke a session token."""
    auth_manager.revoke_session(request.token)
    return {"ok": True}


@router.get("/me")
async def me(authorization: Optional[str] = Header(None)):
    """Return the current user from a Bearer token, or 401."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = auth_manager.validate_session(authorization[7:])
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return user
