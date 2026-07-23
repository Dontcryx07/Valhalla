import { useEffect, useRef, useState } from "react";
import Draggable from "react-draggable";

const SENTIMENT_COLOR = {
  positive: "#51cf66",
  neutral: "#8a8a96",
  negative: "#ff6b6b",
};

export default function ConversationFeed({ conversations, minimized = false, onToggleMinimized }) {
  const nodeRef = useRef(null);
  const [position, setPosition] = useState(() => ({
    x: Math.max(0, window.innerWidth - 306),
    y: 16,
  }));

  // Keep the panel usable after resizing or minimizing, matching agent cards.
  useEffect(() => {
    const clampToViewport = () => {
      const node = nodeRef.current;
      if (!node) return;
      setPosition((previous) => ({
        x: Math.min(Math.max(0, previous.x), Math.max(0, window.innerWidth - node.offsetWidth)),
        y: Math.min(Math.max(0, previous.y), Math.max(0, window.innerHeight - node.offsetHeight)),
      }));
    };
    clampToViewport();
    window.addEventListener("resize", clampToViewport);
    return () => window.removeEventListener("resize", clampToViewport);
  }, [minimized, conversations?.length]);

  if (!conversations || conversations.length === 0) return null;

  return (
    <Draggable
      nodeRef={nodeRef}
      position={position}
      onDrag={(_event, dragData) => setPosition({ x: dragData.x, y: dragData.y })}
      handle=".conversation-feed-drag-handle"
      cancel=".conversation-feed-toggle"
      bounds="parent"
    >
      <div ref={nodeRef} style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: 290,
        maxHeight: minimized ? "none" : "60vh",
        overflowY: minimized ? "hidden" : "auto",
        background: "rgba(0,0,0,0.82)",
        backdropFilter: "blur(10px)",
        border: "1px solid rgba(212,160,74,0.12)",
        borderRadius: 6,
        padding: "10px 12px",
        fontFamily: "'Outfit', sans-serif",
        zIndex: 1000,
      }}>
        <div className="conversation-feed-drag-handle" style={{
          display: "flex",
          alignItems: "center",
          fontSize: 8,
          letterSpacing: 1.2,
          textTransform: "uppercase",
          color: "#6b6b78",
          marginBottom: minimized ? 0 : 8,
          fontFamily: "'Space Mono', monospace",
          cursor: "grab",
        }}>
          <span
            aria-label="Drag conversation feed"
            title="Drag conversation feed"
            style={{ color: "#6b6b78", fontSize: 11, lineHeight: "18px", marginRight: 6 }}
          >
            ≡
          </span>
          <span>Conversation Feed</span>
          <button
            className="conversation-feed-toggle"
            type="button"
            onClick={onToggleMinimized}
            aria-label={minimized ? "Maximize conversation feed" : "Minimize conversation feed"}
            title={minimized ? "Maximize conversation feed" : "Minimize conversation feed"}
            style={{ marginLeft: "auto", width: 20, height: 18, padding: 0, border: "1px solid rgba(212,160,74,0.22)", borderRadius: 3, background: "rgba(212,160,74,0.06)", color: "#d4a04a", cursor: "pointer", fontFamily: "'Space Mono', monospace", fontSize: 13, lineHeight: 1 }}
          >
            {minimized ? "+" : "−"}
          </button>
        </div>
        {!minimized && conversations.map((conversation, index) => (
          <div key={index} style={{
            marginBottom: 9,
            paddingBottom: 9,
            borderBottom: index < conversations.length - 1 ? "1px solid rgba(212,160,74,0.07)" : "none",
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
              <span style={{
                width: 6, height: 6, borderRadius: "50%",
                background: SENTIMENT_COLOR[conversation.sentiment] || "#8a8a96", flexShrink: 0,
              }} />
              <span style={{ fontSize: 12, color: "#eeeef4", fontWeight: 500 }}>
                {(conversation.participants || []).join("  &  ")}
              </span>
              <span style={{ marginLeft: "auto", fontSize: 9, color: "#6b6b78", fontFamily: "'Space Mono', monospace" }}>
                {conversation.time}
              </span>
            </div>
            <div style={{ fontSize: 11, color: "#b8b8c4", lineHeight: 1.4 }}>{conversation.summary}</div>
            {conversation.location && (
              <div style={{ fontSize: 9, color: "#6b6b78", marginTop: 2, fontFamily: "'Space Mono', monospace" }}>
                @ {conversation.location}
              </div>
            )}
          </div>
        ))}
      </div>
    </Draggable>
  );
}
