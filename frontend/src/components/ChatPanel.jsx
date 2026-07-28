import { useEffect, useRef } from "react";
import ChatBubble from "./ChatBubble";

export default function ChatPanel({ conversation, selfId, color, revealedCount }) {
  const bottomRef = useRef(null);
  const messages = conversation?.messages || [];
  const visible = revealedCount != null ? messages.slice(0, revealedCount) : messages;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [visible.length]);

  if (!messages.length) return null;

  return (
    <div style={{
      marginTop: 8,
      borderTop: "1px solid rgba(212,160,74,0.08)",
      paddingTop: 8,
      maxHeight: 180,
      overflowY: "auto",
    }}>
      <div style={{
        fontSize: 8,
        color: "#6b6b78",
        marginBottom: 6,
        textTransform: "uppercase",
        letterSpacing: 1.2,
        fontFamily: "'Space Mono', monospace",
      }}>
        Chat with {conversation.partner_name}
      </div>
      {visible.map((msg, i) => (
        <ChatBubble
          key={i}
          text={msg.text}
          isSelf={msg.speaker === selfId}
          color={color}
        />
      ))}
      {revealedCount != null && visible.length < messages.length && (
        <div style={{
          fontSize: 9,
          color: "#6b6b78",
          textAlign: "center",
          marginTop: 4,
          fontStyle: "italic",
          fontFamily: "'Space Mono', monospace",
        }}>
          ···
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  );
}
