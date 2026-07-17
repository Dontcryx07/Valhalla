export default function ActionDetail({ action }) {
  if (!action) {
    return (
      <div style={{
        fontSize: 11,
        color: "#6b6b78",
        fontStyle: "italic",
        fontFamily: "'Outfit', sans-serif",
        fontWeight: 300,
      }}>
        Idle
      </div>
    );
  }

  const timeRange = action.start_time && action.end_time
    ? `${action.start_time} – ${action.end_time}`
    : null;

  return (
    <div>
      <div style={{
        fontSize: 11,
        color: "#d0d0da",
        lineHeight: 1.4,
        fontFamily: "'Outfit', sans-serif",
        fontWeight: 400,
      }}>
        {action.description}
      </div>
      {timeRange && (
        <div style={{
          fontSize: 9,
          color: "#6b6b78",
          fontFamily: "'Space Mono', monospace",
          marginTop: 2,
        }}>
          {timeRange}
        </div>
      )}
    </div>
  );
}
