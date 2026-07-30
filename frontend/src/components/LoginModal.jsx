import { useState } from "react";

const overlayStyle = {
  position: "fixed",
  inset: 0,
  background: "rgba(0,0,0,0.55)",
  backdropFilter: "blur(4px)",
  zIndex: 2000,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
};

const modalStyle = {
  background: "#111118",
  border: "1px solid rgba(212,160,74,0.25)",
  borderRadius: 8,
  padding: "28px 24px 22px",
  width: 300,
  fontFamily: "'Outfit', sans-serif",
  color: "#d0d0da",
  boxShadow: "0 18px 60px rgba(0,0,0,0.7)",
};

const inputStyle = {
  width: "100%",
  padding: "7px 10px",
  borderRadius: 4,
  border: "1px solid rgba(212,160,74,0.25)",
  background: "#1a1a24",
  color: "#eee",
  fontFamily: "'Space Mono', monospace",
  fontSize: 12,
  outline: "none",
  boxSizing: "border-box",
};

const btnStyle = {
  width: "100%",
  padding: "8px 0",
  borderRadius: 4,
  border: "1px solid rgba(212,160,74,0.35)",
  background: "rgba(212,160,74,0.12)",
  color: "#e7bd70",
  fontFamily: "'Space Mono', monospace",
  fontSize: 11,
  letterSpacing: "0.06em",
  cursor: "pointer",
  textTransform: "uppercase",
};

export default function LoginModal({ onClose, onLogin }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!email || !password) return;
    setBusy(true);
    setError("");
    try {
      await onLogin(email.trim(), password);
      onClose();
    } catch (err) {
      setError(err.message || "Login failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={overlayStyle} onClick={onClose}>
      <form style={modalStyle} onClick={(e) => e.stopPropagation()} onSubmit={handleSubmit}>
        <div
          style={{
            fontSize: 13,
            fontWeight: 600,
            color: "#e7bd70",
            letterSpacing: "0.1em",
            textTransform: "uppercase",
            marginBottom: 18,
          }}
        >
          Admin Login
        </div>
        <label style={{ display: "block", fontSize: 10, color: "#888", letterSpacing: "0.06em", marginBottom: 4 }}>
          EMAIL
        </label>
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="you@college.edu"
          autoFocus
          style={inputStyle}
        />
        <label style={{ display: "block", fontSize: 10, color: "#888", letterSpacing: "0.06em", marginTop: 14, marginBottom: 4 }}>
          PASSWORD
        </label>
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="password"
          style={inputStyle}
        />
        {error && (
          <div style={{ marginTop: 12, padding: "6px 8px", borderRadius: 4, background: "rgba(255,107,107,0.12)", color: "#ff9494", fontSize: 11 }}>
            {error}
          </div>
        )}
        <button type="submit" disabled={busy || !email || !password} style={{ ...btnStyle, marginTop: 18, opacity: busy || !email || !password ? 0.45 : 1 }}>
          {busy ? "LOGGING IN..." : "LOGIN"}
        </button>
        <button type="button" onClick={onClose} style={{ ...btnStyle, marginTop: 8, background: "transparent", border: "1px solid rgba(255,255,255,0.08)", color: "#777" }}>
          CANCEL
        </button>
      </form>
    </div>
  );
}
