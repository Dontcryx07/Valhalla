const CATEGORY_COLOR = {
  "technical-cultural": "#5b9bd5",
  sports: "#51cf66",
  academic: "#d4a04a",
};

function EventCard({ event, active }) {
  const color = CATEGORY_COLOR[event.category] || "#8a8a96";
  return (
    <div style={{
      padding: "8px 0", borderBottom: "1px solid rgba(212,160,74,0.07)",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <span style={{ width: 6, height: 6, borderRadius: "50%", background: color, boxShadow: active ? `0 0 10px ${color}` : "none" }} />
        <span style={{ fontSize: 11, color: "#eeeef4", fontWeight: 500 }}>{event.name}</span>
      </div>
      <div style={{ fontFamily: "'Space Mono', monospace", color: "#8a8a96", fontSize: 8, marginTop: 3 }}>
        {active ? "LIVE" : `${event.start_time}–${event.end_time}`} · {event.location_id}
      </div>
      <div style={{ color: "#b8b8c4", fontSize: 10, lineHeight: 1.35, marginTop: 3 }}>{event.description}</div>
      <div style={{ color, fontSize: 8, fontFamily: "'Space Mono', monospace", marginTop: 4 }}>
        {event.attendance_count}/{event.capacity} attending
      </div>
    </div>
  );
}

export default function EventsPanel({ events }) {
  const active = events?.active || [];
  const upcoming = events?.upcoming || [];
  if (!active.length && !upcoming.length) return null;
  return (
    <aside style={{
      position: "fixed", right: 16, bottom: 60, width: 290, maxHeight: "34vh", overflowY: "auto",
      background: "rgba(7,10,12,0.90)", backdropFilter: "blur(10px)",
      border: "1px solid rgba(91,155,213,0.18)", borderRadius: 6, padding: "9px 12px",
      fontFamily: "'Outfit', sans-serif", zIndex: 900,
    }}>
      <div style={{ color: "#6f9fd1", fontFamily: "'Space Mono', monospace", fontSize: 8, letterSpacing: 1.2, textTransform: "uppercase" }}>
        Campus pulse
      </div>
      {active.map((event) => <EventCard key={event.id} event={event} active />)}
      {upcoming.map((event) => <EventCard key={event.id} event={event} active={false} />)}
    </aside>
  );
}
