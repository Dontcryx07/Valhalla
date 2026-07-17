// Color legend: maps each agent's dot color to their name. Click to focus.
export default function Legend({ agents, focusedId, onFocus }) {
  if (!agents) return null;
  const entries = Object.entries(agents);
  if (!entries.length) return null;

  return (
    <div style={{
      position: "fixed",
      top: 16,
      left: 16,
      background: "rgba(0,0,0,0.80)",
      backdropFilter: "blur(8px)",
      border: "1px solid rgba(212,160,74,0.12)",
      borderRadius: 6,
      padding: "8px 10px",
      fontFamily: "'Outfit', sans-serif",
      fontSize: 12,
      zIndex: 1000,
      minWidth: 150,
    }}>
      <div style={{
        fontSize: 8,
        letterSpacing: 1.2,
        textTransform: "uppercase",
        color: "#6b6b78",
        marginBottom: 6,
        fontFamily: "'Space Mono', monospace",
      }}>
        Students
      </div>
      {entries.map(([id, a]) => {
        const focused = id === focusedId;
        return (
          <div
            key={id}
            onClick={() => onFocus && onFocus(focused ? null : id)}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              padding: "3px 4px",
              borderRadius: 4,
              cursor: "pointer",
              background: focused ? "rgba(212,160,74,0.14)" : "transparent",
              transition: "background 0.15s",
            }}
          >
            <span style={{
              width: 9, height: 9, borderRadius: "50%",
              background: a.color, flexShrink: 0,
              boxShadow: focused ? `0 0 8px ${a.color}` : "none",
            }} />
            <span style={{ color: focused ? "#eeeef4" : "#c8c8d2", fontWeight: focused ? 600 : 400 }}>
              {a.name}
            </span>
            {a.in_conversation && (
              <span style={{ marginLeft: "auto", fontSize: 9, color: "#d4a04a" }}>chatting</span>
            )}
          </div>
        );
      })}
    </div>
  );
}
