import { useState, useEffect } from "react";
import useSimState from "./hooks/useSimState";
import SimCanvas from "./components/SimCanvas";
import AgentWindow from "./components/AgentWindow";
import InfoBar from "./components/InfoBar";
import Legend from "./components/Legend";
import ConversationFeed from "./components/ConversationFeed";
import "./App.css";

function computeScatter(index, total) {
  const cols = 2;
  const col = index % cols;
  const row = Math.floor(index / cols);
  const x = 20 + col * 280;
  const y = 60 + row * 240;
  return { x, y };
}

export default function App() {
  const snapshot = useSimState();
  const [agentMap, setAgentMap] = useState(null);
  const [placed, setPlaced] = useState({});
  const [day, setDay] = useState(0);
  const [focusedId, setFocusedId] = useState(null);

  useEffect(() => {
    if (!snapshot) return;
    if (snapshot.type === "day_reset" || snapshot.type === "reset") {
      setAgentMap(null);
      setPlaced({});
      return;
    }
    if (snapshot.agents) {
      setAgentMap(snapshot.agents);
      if (snapshot.day !== undefined) setDay(snapshot.day);
    }
  }, [snapshot]);

  const agentIds = agentMap ? Object.keys(agentMap) : [];

  return (
    <div className="app-root">
      <SimCanvas snapshot={snapshot} focusedId={focusedId} onFocus={setFocusedId} />

      <Legend agents={agentMap} focusedId={focusedId} onFocus={setFocusedId} />
      <ConversationFeed conversations={snapshot?.recent_conversations} />

      {agentIds.map((id, i) => {
        const data = agentMap[id];
        const pos = placed[id] || computeScatter(i, agentIds.length);
        return (
          <AgentWindow
            key={`${id}-${day}`}
            agentId={id}
            data={data}
            speed={snapshot?.speed}
            defaultPosition={{ x: pos.x, y: pos.y }}
            focused={id === focusedId}
            onFocus={() => setFocusedId(id === focusedId ? null : id)}
          />
        );
      })}

      <InfoBar snapshot={snapshot} />
    </div>
  );
}
