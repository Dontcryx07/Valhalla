export default function InfoBar({ snapshot }) {
  if (!snapshot) return null;
  const { tick, time, day, agents: agentMap, speed } = snapshot;
  const agentCount = agentMap ? Object.keys(agentMap).length : 0;
  const pausedCount = agentMap
    ? Object.values(agentMap).filter((a) => a.paused).length
    : 0;
  const chattingCount = agentMap
    ? Object.values(agentMap).filter((a) => a.conversation).length
    : 0;

  return (
    <div style={{
      position: "fixed",
      bottom: 16,
      left: 16,
      background: "rgba(0,0,0,0.80)",
      backdropFilter: "blur(8px)",
      border: "1px solid rgba(212,160,74,0.12)",
      borderRadius: 6,
      padding: "8px 14px",
      fontFamily: "'Space Mono', monospace",
      fontSize: 11,
      pointerEvents: "none",
      display: "flex",
      gap: 14,
      zIndex: 1000,
      letterSpacing: 0.3,
    }}>
      <span style={{ color: "#6b6b78" }}>TICK</span>
      <span style={{ color: "#d0d0da" }}>{tick}</span>
      <span style={{ color: "#6b6b78" }}>|</span>
      <span style={{ color: "#6b6b78" }}>TIME</span>
      <span style={{ color: "#d4a04a", fontWeight: 700 }}>{time}</span>
      <span style={{ color: "#6b6b78" }}>|</span>
      <span style={{ color: "#6b6b78" }}>DAY</span>
      <span style={{ color: "#d0d0da" }}>{day}</span>
      {speed?.real_min_per_day && (
        <>
          <span style={{ color: "#6b6b78" }}>|</span>
          <span style={{ color: "#6b6b78" }}>PACE</span>
          <span style={{ color: "#d0d0da" }}>{speed.real_min_per_day}m/day</span>
        </>
      )}
      <span style={{ color: "#6b6b78" }}>|</span>
      <span style={{ color: "#d0d0da" }}>{agentCount}</span>
      <span style={{ color: "#6b6b78" }}>AGENTS</span>
      {pausedCount > 0 && (
        <>
          <span style={{ color: "#6b6b78" }}>|</span>
          <span style={{ color: "#5b7db5" }}>{pausedCount} PAUSED</span>
        </>
      )}
      {chattingCount > 0 && (
        <>
          <span style={{ color: "#6b6b78" }}>|</span>
          <span style={{ color: "#d4a04a" }}>{chattingCount} CHATTING</span>
        </>
      )}
    </div>
  );
}
