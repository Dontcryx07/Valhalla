import { useRef, useState, useEffect } from "react";
import Draggable from "react-draggable";
import WindowHeader from "./WindowHeader";
import ActionDetail from "./ActionDetail";
import ChatPanel from "./ChatPanel";

function emotionLabel(value) {
  if (value == null) return "Neutral";
  if (value >= 0.9) return "Extremely Joyful";
  if (value >= 0.7) return "Very Happy";
  if (value >= 0.5) return "Happy";
  if (value >= 0.3) return "Neutral";
  if (value >= 0.1) return "Sad";
  return "Extremely Sad";
}

export default function AgentWindow({ agentId, data, speed, defaultPosition, focused, onFocus }) {
  const nodeRef = useRef(null);
  const [minimized, setMinimized] = useState(false);
  const [history, setHistory] = useState([]);
  const [revealedCounts, setRevealedCounts] = useState({});

  const action = data.current_action;
  const conversation = data.conversation;
  const paused = data.paused;

  // Accumulate conversation history
  useEffect(() => {
    if (!conversation) return;
    setHistory((prev) => {
      const last = prev[prev.length - 1];
      if (last && last.partner_name === conversation.partner_name) {
        const lastMsg = last.messages[last.messages.length - 1];
        const newMsg = conversation.messages[conversation.messages.length - 1];
        if (lastMsg && newMsg && lastMsg.text === newMsg.text) return prev;
      }
      return [...prev, { ...conversation, timestamp: Date.now() }];
    });
  }, [conversation]);

  // Staged message reveal: when a new conversation arrives, reveal messages
  // one-by-one based on duration / message count.
  const realMsPerSimMinute = speed?.real_ms_per_sim_minute || 40000;
  useEffect(() => {
    if (!conversation?.messages?.length || !conversation.duration_minutes) return;
    const key = `${conversation.partner_name}_${conversation.messages.length}_${Date.now()}`;
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
  }, [conversation?.messages?.length, conversation?.duration_minutes]);

  const locationLabel = data.position?.location_id || null;
  const pos = data.position
    ? `(${Math.round(data.position.x)}, ${Math.round(data.position.y)})`
    : null;

  return (
    <Draggable
      nodeRef={nodeRef}
      defaultPosition={defaultPosition}
      handle=".window-handle"
      bounds="parent"
    >
      <div
        ref={nodeRef}
        style={{
          position: "absolute",
          width: 260,
          background: "rgba(10,10,10,0.88)",
          backdropFilter: "blur(14px)",
          WebkitBackdropFilter: "blur(14px)",
          border: focused
            ? `1px solid ${data.color}`
            : paused
            ? `1px solid ${data.color}40`
            : "1px solid rgba(212,160,74,0.10)",
          borderRadius: 6,
          padding: "10px 12px",
          color: "#d0d0da",
          fontFamily: "'Outfit', sans-serif",
          userSelect: "none",
          zIndex: focused ? 200 : paused ? 100 : 10,
          boxShadow: focused
            ? `0 0 28px ${data.color}55, 0 4px 24px rgba(0,0,0,0.5)`
            : paused
            ? `0 0 24px ${data.color}25, 0 4px 24px rgba(0,0,0,0.5)`
            : "0 2px 16px rgba(0,0,0,0.4)",
          transition: "box-shadow 0.3s, border-color 0.3s",
        }}
      >
        <div className="window-handle" style={{ cursor: "grab" }} onClick={onFocus}>
          <WindowHeader
            name={data.name}
            color={data.color}
            actionType={action?.action_type}
          />
          <button
            onClick={() => setMinimized(!minimized)}
            style={{
              position: "absolute",
              top: 6,
              right: 8,
              background: "none",
              border: "none",
              color: "#6b6b78",
              cursor: "pointer",
              fontSize: 13,
              fontFamily: "'Space Mono', monospace",
              transition: "color 0.2s",
            }}
            onMouseEnter={(e) => e.target.style.color = "#d4a04a"}
            onMouseLeave={(e) => e.target.style.color = "#6b6b78"}
          >
            {minimized ? "+" : "–"}
          </button>
        </div>

        {!minimized && (
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

            {/* Conversation history with staged reveal */}
            {history.map((conv, i) => {
              const convKey = conv.timestamp || i;
              const revealed = revealedCounts[convKey] ?? conv.messages?.length ?? 0;
              return (
                <div key={convKey}>
                  {i > 0 && (
                    <div style={{
                      height: 1,
                      background: "rgba(212,160,74,0.08)",
                      margin: "8px 0",
                    }} />
                  )}
                  <ChatPanel
                    conversation={conv}
                    selfId={data.name}
                    color={data.color}
                    revealedCount={revealed}
                  />
                </div>
              );
            })}
          </>
        )}
      </div>
    </Draggable>
  );
}
