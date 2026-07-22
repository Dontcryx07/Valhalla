export default function DebugPanel({ health }) {
  if (!health) return null;
  return (
    <aside style={{
      position: "fixed", left: 16, top: 120, width: 255, background: "rgba(8,10,12,0.92)",
      border: `1px solid ${health.healthy ? "rgba(81,207,102,.25)" : "rgba(255,107,107,.5)"}`,
      borderRadius: 6, padding: "10px 12px", zIndex: 1200, fontFamily: "'Space Mono', monospace", fontSize: 9,
    }}>
      <div style={{ color: health.healthy ? "#51cf66" : "#ff6b6b", letterSpacing: 1, marginBottom: 7 }}>
        {health.healthy ? "SYSTEM HEALTHY" : "HEALTH WARNINGS"}
      </div>
      <div style={{ color: "#a9a9b2", lineHeight: 1.7 }}>
        tick {health.tick} · {health.agents} agents<br />
        moving {health.moving} · paused {health.paused}<br />
        conversations {health.conversations} · tasks {health.event_loop_tasks}
      </div>
      {!!health.anomalies?.length && <div style={{ color: "#ff8e8e", marginTop: 7 }}>
        {health.anomalies.map((item, index) => <div key={`${item.agent_id}-${index}`}>{item.agent_id}: {item.kind}</div>)}
      </div>}
    </aside>
  );
}
