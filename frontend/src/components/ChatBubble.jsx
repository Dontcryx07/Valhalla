export default function ChatBubble({ text, isSelf, color }) {
  return (
    <div style={{
      display: "flex",
      justifyContent: isSelf ? "flex-end" : "flex-start",
      marginBottom: 4,
      animation: "fadeIn 0.2s ease-out",
    }}>
      <div style={{
        maxWidth: "80%",
        padding: "5px 9px",
        borderRadius: 8,
        borderBottomRightRadius: isSelf ? 2 : 8,
        borderBottomLeftRadius: isSelf ? 8 : 2,
        background: isSelf ? color : "rgba(255,255,255,0.06)",
        color: isSelf ? "#09090e" : "#d0d0da",
        fontSize: 11,
        lineHeight: 1.35,
        wordBreak: "break-word",
        fontFamily: "'Outfit', sans-serif",
        fontWeight: isSelf ? 500 : 400,
      }}>
        {text}
      </div>
    </div>
  );
}
