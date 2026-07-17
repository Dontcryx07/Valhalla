// Campus-wide conversation feed: the most recent conversations as they happen.
const SENTIMENT_COLOR = {
  positive: "#51cf66",
  neutral: "#8a8a96",
  negative: "#ff6b6b",
};

export default function ConversationFeed({ conversations }) {
  if (!conversations || conversations.length === 0) return null;

  return (
    <div style={{
      position: "fixed",
      top: 16,
      right: 16,
      width: 290,
      maxHeight: "60vh",
      overflowY: "auto",
      background: "rgba(0,0,0,0.82)",
      backdropFilter: "blur(10px)",
      border: "1px solid rgba(212,160,74,0.12)",
      borderRadius: 6,
      padding: "10px 12px",
      fontFamily: "'Outfit', sans-serif",
      zIndex: 1000,
    }}>
      <div style={{
        fontSize: 8,
        letterSpacing: 1.2,
        textTransform: "uppercase",
        color: "#6b6b78",
        marginBottom: 8,
        fontFamily: "'Space Mono', monospace",
      }}>
        Conversation Feed
      </div>
      {conversations.map((c, i) => (
        <div key={i} style={{
          marginBottom: 9,
          paddingBottom: 9,
          borderBottom: i < conversations.length - 1 ? "1px solid rgba(212,160,74,0.07)" : "none",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
            <span style={{
              width: 6, height: 6, borderRadius: "50%",
              background: SENTIMENT_COLOR[c.sentiment] || "#8a8a96", flexShrink: 0,
            }} />
            <span style={{ fontSize: 12, color: "#eeeef4", fontWeight: 500 }}>
              {(c.participants || []).join("  &  ")}
            </span>
            <span style={{
              marginLeft: "auto", fontSize: 9, color: "#6b6b78",
              fontFamily: "'Space Mono', monospace",
            }}>
              {c.time}
            </span>
          </div>
          <div style={{ fontSize: 11, color: "#b8b8c4", lineHeight: 1.4 }}>
            {c.summary}
          </div>
          {c.location && (
            <div style={{
              fontSize: 9, color: "#6b6b78", marginTop: 2,
              fontFamily: "'Space Mono', monospace",
            }}>
              @ {c.location}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
