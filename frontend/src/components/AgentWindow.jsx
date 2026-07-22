import { useRef, useState, useEffect } from "react";
import Draggable from "react-draggable";
import WindowHeader from "./WindowHeader";
import ActionDetail from "./ActionDetail";
import ChatPanel from "./ChatPanel";

function emotionLabel(value) {
  if (value == null) return "Neutral";
  if (value >= 0.9) return "Extremely Joyful";
  if (value >= 0.7) return "Very Happy";
  if (value > 0.5) return "Happy";
  if (value >= 0.3) return "Neutral";
  if (value >= 0.1) return "Sad";
  return "Extremely Sad";
}

export default function AgentWindow({ agentId, data, speed, defaultPosition, expanded = false, expandedLayer = 0, onToggle }) {
  const nodeRef = useRef(null);
  const [position, setPosition] = useState(defaultPosition);
  const [revealedCounts, setRevealedCounts] = useState({});

  const action = data.current_action;
  const conversation = data.conversation;
  const paused = data.paused;
  const conversationId = conversation
    ? `${conversation.partner_id || conversation.partner_name}_${conversation.started_tick ?? "pending"}`
    : null;
  // A transcript belongs to the live conversation state, not to the agent
  // card's lifetime.  The backend clears this state when the simulated chat
  // finishes; rendering only an active/generating conversation automatically
  // collapses the chat panel as the agent starts their next task.
  const showConversation = Boolean(
    conversation?.messages?.length
      && ["generating", "active"].includes(conversation.status)
  );

  // Staged message reveal: when a new conversation arrives, reveal messages
  // one-by-one based on duration / message count.
  const realMsPerSimMinute = speed?.real_ms_per_sim_minute || 40000;
  useEffect(() => {
    if (!conversation?.messages?.length || !conversation.duration_minutes) return;
    const key = conversationId;
    const msgs = conversation.messages.length;
    const totalMs = conversation.duration_minutes * realMsPerSimMinute;
    const delayPerMsg = totalMs / msgs;

    setRevealedCounts((prev) => ({ ...prev, [key]: 0 }));

    let i = 0;
    const timer = setInterval(() => {
      i++;
      setRevealedCounts((prev) => {
        if ((prev[key] || 0) >= msgs) return prev;
        return { ...prev, [key]: i };
      });
      if (i >= msgs) clearInterval(timer);
    }, delayPerMsg);

    return () => clearInterval(timer);
  }, [conversationId, conversation?.messages?.length, conversation?.duration_minutes]);

  const locationLabel = action?.action_type === "move"
    ? `EN ROUTE → ${action.location_id || "destination"}`
    : data.position?.location_id || null;
  const pos = data.position
    ? `(${Math.round(data.position.x)}, ${Math.round(data.position.y)})`
    : null;

  // `react-draggable` bounds movement using the card's size when the drag
  // starts. A compact card can therefore be placed near an edge and become
  // partly off-screen when its details increase the size. Clamp the saved
  // position after every expand/collapse transition as a second boundary.
  useEffect(() => {
    const node = nodeRef.current;
    if (!node) return;
    const maxX = Math.max(0, window.innerWidth - node.offsetWidth);
    const maxY = Math.max(0, window.innerHeight - node.offsetHeight);
    setPosition((previous) => ({
      x: Math.min(Math.max(0, previous.x), maxX),
      y: Math.min(Math.max(0, previous.y), maxY),
    }));
  }, [expanded]);

  return (
    <Draggable
      nodeRef={nodeRef}
      position={position}
      onDrag={(_event, dragData) => setPosition({ x: dragData.x, y: dragData.y })}
      handle=".agent-window-drag-handle"
      cancel=".agent-window-toggle, .agent-window-title"
      bounds="parent"
    >
      <div
        ref={nodeRef}
        style={{
          position: "absolute",
          width: expanded ? 260 : 214,
          background: "rgba(10,10,10,0.88)",
          backdropFilter: "blur(14px)",
          WebkitBackdropFilter: "blur(14px)",
          border: expanded
            ? `1px solid ${data.color}`
            : paused
            ? `1px solid ${data.color}40`
            : "1px solid rgba(212,160,74,0.10)",
          borderRadius: 6,
          padding: "10px 12px",
          color: "#d0d0da",
          fontFamily: "'Outfit', sans-serif",
          userSelect: "none",
          zIndex: expanded ? 300 + expandedLayer : paused ? 100 : 10,
          boxShadow: expanded
            ? `0 0 28px ${data.color}55, 0 4px 24px rgba(0,0,0,0.5)`
            : paused
            ? `0 0 24px ${data.color}25, 0 4px 24px rgba(0,0,0,0.5)`
            : "0 2px 16px rgba(0,0,0,0.4)",
          transition: "box-shadow 0.3s, border-color 0.3s",
        }}
      >
        <div className="agent-window-drag-handle" style={{ display: "flex", alignItems: "flex-start", gap: 6, minHeight: 22, cursor: "grab" }}>
          <span
            className="agent-window-drag-grip"
            aria-label={`Drag ${data.name}'s inspector`}
            title="Drag inspector"
            style={{ color: "#6b6b78", fontFamily: "'Space Mono', monospace", fontSize: 11, lineHeight: "20px" }}
          >
            ≡
          </span>
          <button
            className="agent-window-title"
            type="button"
            onClick={onToggle}
            aria-expanded={expanded}
            title={expanded ? "Collapse details" : "Show details"}
            style={{
              flex: 1, minWidth: 0, border: 0, padding: 0, margin: 0,
              background: "transparent", textAlign: "left", cursor: "pointer", color: "inherit",
            }}
          >
            <WindowHeader name={data.name} color={data.color} />
          </button>
          <button
            className="agent-window-toggle"
            type="button"
            onClick={(event) => {
              event.stopPropagation();
              onToggle();
            }}
            aria-label={expanded ? `Collapse ${data.name}'s details` : `Show ${data.name}'s details`}
            aria-expanded={expanded}
            title={expanded ? "Collapse details" : "Show details"}
            style={{
              flex: "0 0 auto",
              marginTop: 0,
              background: "rgba(212,160,74,0.06)",
              border: "1px solid rgba(212,160,74,0.22)",
              borderRadius: 3,
              color: "#d4a04a",
              cursor: "pointer",
              fontSize: 13,
              fontFamily: "'Space Mono', monospace",
              transition: "color 0.2s",
              width: 20,
              height: 18,
              lineHeight: 1,
            }}
          >
            {expanded ? "−" : "+"}
          </button>
        </div>

        {!expanded ? (
          <div style={{
            color: "#9b9ba8", fontSize: 10, lineHeight: 1.35, paddingRight: 5,
            whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
          }} title={`${data.position?.location_id || "Unknown"} — ${data.activity || "Idle"}`}>
            <span style={{ color: "#6b6b78", fontFamily: "'Space Mono', monospace", fontSize: 8 }}>
              {data.position?.location_id || "UNKNOWN"}
            </span>
            <span style={{ color: "#5a5a66", margin: "0 5px" }}>·</span>
            {data.activity || "Idle"}
          </div>
        ) : (
          <>
            {/* Location */}
            <div style={{
              fontSize: 9,
              fontFamily: "'Space Mono', monospace",
              color: "#6b6b78",
              marginBottom: 4,
              letterSpacing: 0.3,
            }}>
              {locationLabel ? `${locationLabel}  ${pos}` : pos}
            </div>

            <ActionDetail action={action} />

            {action?.event_id && (
              <div style={{
                marginTop: 5, color: "#6f9fd1", fontSize: 8, letterSpacing: 0.7,
                fontFamily: "'Space Mono', monospace", textTransform: "uppercase",
              }}>
                Campus event · {action.event_id}
              </div>
            )}

            {/* Energy bar */}
            <div style={{ marginBottom: 4 }}>
              <div style={{ fontSize: 9, color: "#6b6b78", fontFamily: "'Space Mono', monospace", marginBottom: 2 }}>
                ENERGY
              </div>
              <div style={{ height: 6, background: "rgba(255,255,255,0.06)", borderRadius: 3, overflow: "hidden" }}>
                <div style={{
                  height: "100%",
                  width: `${Math.round((data.energy_level || 0) * 100)}%`,
                  background: "linear-gradient(90deg, #ff6b6b, #ffd43b, #51cf66)",
                  borderRadius: 3,
                  transition: "width 0.3s",
                }} />
              </div>
            </div>

            {/* Emotion label */}
            <div style={{ fontSize: 9, color: "#6b6b78", fontFamily: "'Space Mono', monospace", marginBottom: 6 }}>
              EMOTION: <span style={{ color: "#d0d0da" }}>{emotionLabel(data.emotion_state)}</span>
            </div>

            {paused && !conversation && (
              <div style={{
                fontSize: 9,
                color: data.color,
                marginTop: 4,
                fontStyle: "italic",
                letterSpacing: 0.5,
                textTransform: "uppercase",
              }}>
                Paused
              </div>
            )}

            {/* The panel unmounts when the next non-chat action starts. */}
            {showConversation && (
              <ChatPanel
                conversation={conversation}
                selfId={data.name}
                color={data.color}
                revealedCount={revealedCounts[conversationId] ?? conversation.messages.length}
              />
            )}
          </>
        )}
      </div>
    </Draggable>
  );
}
