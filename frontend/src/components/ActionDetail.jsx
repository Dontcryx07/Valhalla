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
  const routeProgress = action.action_type === "move" && action.path?.length > 1
    ? Math.round(((action.path_index || 0) / (action.path.length - 1)) * 100)
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
      {routeProgress != null && (
        <div style={{ fontSize: 8, color: "#5b9bd5", fontFamily: "'Space Mono', monospace", marginTop: 3 }}>
          ROUTE {Math.max(0, Math.min(100, routeProgress))}%
        </div>
      )}
    </div>
  );
}
