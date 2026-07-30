import { useState } from "react";
import { useAuth } from "../hooks/useAuth";
import LoginModal from "./LoginModal";

export default function LoginButton() {
  const { user, isAuthenticated, loading, login, logout } = useAuth();
  const [showModal, setShowModal] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);

  if (loading) return null;

  const basePill = {
    position: "fixed",
    bottom: 16,
    right: 16,
    zIndex: 1500,
    fontFamily: "'Space Mono', monospace",
    fontSize: 11,
    fontWeight: 600,
    letterSpacing: "0.08em",
    borderRadius: 6,
    padding: "7px 16px",
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    gap: 7,
    textTransform: "uppercase",
    transition: "all 0.15s",
  };

  if (!isAuthenticated) {
    return (
      <>
        <button
          style={{
            ...basePill,
            background: "#1a1a28",
            border: "1px solid rgba(212,160,74,0.45)",
            color: "#e7bd70",
            boxShadow: "0 2px 12px rgba(0,0,0,0.5), inset 0 1px 0 rgba(212,160,74,0.15)",
          }}
          onClick={() => setShowModal(true)}
          title="Admin login"
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M15 3h4a2 2 0 012 2v14a2 2 0 01-2 2h-4" />
            <polyline points="10 17 15 12 10 7" />
            <line x1="15" y1="12" x2="3" y2="12" />
          </svg>
          LOGIN
        </button>
        {showModal && <LoginModal onClose={() => setShowModal(false)} onLogin={login} />}
      </>
    );
  }

  return (
    <div style={{ position: "fixed", bottom: 16, right: 16, zIndex: 1500 }}>
      <button
        style={{
          ...basePill,
          background: "rgba(212,160,74,0.12)",
          border: "1px solid rgba(212,160,74,0.35)",
          color: "#e7bd70",
          boxShadow: "0 2px 10px rgba(0,0,0,0.4)",
        }}
        onClick={() => setShowDropdown((v) => !v)}
        title="Logged in — click to logout"
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="8" r="5" />
          <path d="M20 21a8 8 0 00-16 0" />
        </svg>
        {user?.name || user?.email}
      </button>
      {showDropdown && (
        <div
          style={{
            position: "absolute",
            bottom: 42,
            right: 0,
            background: "#111118",
            border: "1px solid rgba(212,160,74,0.25)",
            borderRadius: 6,
            padding: "8px 0",
            minWidth: 170,
            boxShadow: "0 8px 30px rgba(0,0,0,0.6)",
          }}
          onMouseLeave={() => setShowDropdown(false)}
        >
          <div style={{ padding: "6px 14px", fontSize: 10, color: "#888", fontFamily: "'Space Mono', monospace" }}>
            {user?.email}
          </div>
          <div style={{ height: 1, background: "rgba(255,255,255,0.08)", margin: "2px 0" }} />
          <button
            onClick={() => {
              setShowDropdown(false);
              logout();
            }}
            style={{
              display: "block",
              width: "100%",
              textAlign: "left",
              padding: "8px 14px",
              background: "none",
              border: "none",
              color: "#ff9494",
              cursor: "pointer",
              fontSize: 11,
              fontFamily: "'Space Mono', monospace",
            }}
          >
            LOGOUT
          </button>
        </div>
      )}
    </div>
  );
}
