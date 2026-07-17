export default function WindowHeader({ name, color, actionType }) {
  const badgeColors = {
    move: "#2a7de1",
    misc: "#6b6b78",
    conversation: "#d4a04a",
  };

  return (
    <div style={{
      display: "flex",
      alignItems: "center",
      gap: 8,
      marginBottom: 6,
    }}>
      <div style={{
        width: 8,
        height: 8,
        borderRadius: "50%",
        background: color,
        flexShrink: 0,
        boxShadow: `0 0 6px ${color}60`,
      }} />
      <span style={{
        fontSize: 13,
        fontWeight: 500,
        color: "#eeeef4",
        fontFamily: "'Outfit', sans-serif",
        letterSpacing: 0.2,
      }}>
        {name}
      </span>
      {actionType && (
        <span style={{
          fontSize: 8,
          fontWeight: 600,
          color: "#09090e",
          background: badgeColors[actionType] || "#6b6b78",
          padding: "2px 6px",
          borderRadius: 2,
          textTransform: "uppercase",
          letterSpacing: 0.8,
          marginLeft: "auto",
          fontFamily: "'Space Mono', monospace",
        }}>
          {actionType}
        </span>
      )}
    </div>
  );
}
