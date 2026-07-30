import { useState, useEffect, useCallback } from "react";
import useSimState from "./hooks/useSimState";
import { AuthProvider, useAuth } from "./hooks/useAuth";
import SimCanvas from "./components/SimCanvas";
import AgentWindow from "./components/AgentWindow";
import InfoBar from "./components/InfoBar";
import ConversationFeed from "./components/ConversationFeed";
import EventsPanel from "./components/EventsPanel";
import DebugPanel from "./components/DebugPanel";
import RosterManager from "./components/RosterManager";
import LoginButton from "./components/LoginButton";
import { apiUrl } from "./utils/api";
import "./App.css";

function compactTabPosition(index) {
  const side = index % 2;
  const row = Math.floor(index / 2);
  return {
    // Reserve the full inspector width even while this card is compact. A
    // right-hand card can then expand without its controls leaving the view.
    x: side ? Math.max(16, window.innerWidth - 276) : 16,
    y: 66 + row * 92,
  };
}

export default function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}

function AppContent() {
  const { token, isAuthenticated } = useAuth();
  const snapshot = useSimState();
  const [renderSnapshot, setRenderSnapshot] = useState(null);
  const [agentMap, setAgentMap] = useState(null);
  const [focusedId, setFocusedId] = useState(null);
  const [expandedAgentIds, setExpandedAgentIds] = useState(() => new Set());
  const [showDebug, setShowDebug] = useState(false);
  const [conversationFeedMinimized, setConversationFeedMinimized] = useState(false);
  const [controlError, setControlError] = useState(null);
  const [simulationRunning, setSimulationRunning] = useState(true);
  const [rosterOpen, setRosterOpen] = useState(false);

  useEffect(() => {
    if (!snapshot) return;
    // A handoff packet is a transient status message without `agents`.
    // Preserve the latest cards until the next full state arrives.
    if (snapshot.type === "day_reset" || snapshot.type === "reset") {
      setAgentMap(null);
      setFocusedId(null);
      setExpandedAgentIds(new Set());
      return;
    }
    if (snapshot.agents) {
      setAgentMap(snapshot.agents);
    }
    // Handoff status packets intentionally omit a full world frame. Keep the
    // last complete frame rendered until the first new-day frame arrives.
    if (snapshot.tick != null && snapshot.time != null && snapshot.day != null) {
      setRenderSnapshot(snapshot);
    }
    if (typeof snapshot.simulation?.running === "boolean") {
      setSimulationRunning(snapshot.simulation.running);
    }
  }, [snapshot]);

  const providerFailure = snapshot?.status === "provider_failure" ? snapshot.failure : null;
  const simulationError = (snapshot?.status === "error" || providerFailure) ? snapshot.message : null;
  const agentIds = agentMap ? Object.keys(agentMap) : [];

  function toggleAgentInspector(agentId) {
    const isExpanded = expandedAgentIds.has(agentId);
    setExpandedAgentIds((previous) => {
      const next = new Set(previous);
      if (next.has(agentId)) next.delete(agentId);
      else next.add(agentId);
      return next;
    });
    if (isExpanded && focusedId === agentId) setFocusedId(null);
    else if (!isExpanded) setFocusedId(agentId);
  }

  const focusFromMap = useCallback((agentId) => {
    setFocusedId(agentId);
    if (agentId) {
      setExpandedAgentIds((previous) => {
        if (previous.has(agentId)) return previous;
        const next = new Set(previous);
        next.add(agentId);
        return next;
      });
    }
  }, []);

  async function sendControl(path, body) {
    try {
      setControlError(null);
      const headers = { "Content-Type": "application/json" };
      if (token) headers["Authorization"] = `Bearer ${token}`;
      const response = await fetch(apiUrl(path), {
        method: "POST",
        headers,
        body: body ? JSON.stringify(body) : undefined,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || data.detail || "Simulation control failed.");
      return data;
    } catch (error) {
      setControlError(error.message);
      return null;
    }
  }

  return (
    <div className="app-root">
      <SimCanvas snapshot={renderSnapshot || snapshot} focusedId={focusedId} onFocus={focusFromMap} />

      {simulationError && (
        <section className="simulation-error" role="alert" aria-live="assertive">
          <span className="simulation-error__eyebrow">{providerFailure?.title || "Simulation unavailable"}</span>
          <p>{simulationError}</p>
          <small>{providerFailure?.guidance || "Runtime data was left unchanged. Start a fresh simulation when you are ready."}</small>
        </section>
      )}
      {controlError && (
        <section className="simulation-error simulation-error--control" role="status">
          <span className="simulation-error__eyebrow">Timeline control</span>
          <p>{controlError}</p>
        </section>
      )}

      <ConversationFeed
        conversations={snapshot?.recent_conversations}
        minimized={conversationFeedMinimized}
        onToggleMinimized={() => setConversationFeedMinimized((value) => !value)}
      />
      <EventsPanel events={snapshot?.events} />
      {showDebug && <DebugPanel health={snapshot?.health} />}
      <RosterManager open={rosterOpen} onClose={() => setRosterOpen(false)} simulationRunning={simulationRunning} onError={setControlError} isAuthenticated={isAuthenticated} />

      {agentIds.map((id, index) => {
        const expanded = expandedAgentIds.has(id);
        const expandedLayer = [...expandedAgentIds].indexOf(id);
        return (
          <AgentWindow
            key={id}
            agentId={id}
            data={agentMap[id]}
            speed={snapshot?.speed}
            defaultPosition={compactTabPosition(index)}
            expanded={expanded}
            expandedLayer={expandedLayer}
            onToggle={() => toggleAgentInspector(id)}
          />
        );
      })}

      <InfoBar
        snapshot={renderSnapshot || snapshot}
        showDebug={showDebug}
        onToggleDebug={() => setShowDebug((value) => !value)}
        onFastForward={() => sendControl("/api/sim/fast-forward")}
        onSlowDown={() => sendControl("/api/sim/slow-down")}
        onRewind={(rewind) => sendControl("/api/sim/rewind", rewind)}
        simulationRunning={simulationRunning}
        onToggleSimulation={async () => {
          const result = await sendControl(simulationRunning ? "/api/sim/stop" : "/api/sim/start");
          if (result && typeof result.running === "boolean") setSimulationRunning(result.running);
        }}
        onToggleRoster={() => setRosterOpen((value) => !value)}
        isAuthenticated={isAuthenticated}
      />
      <LoginButton />
    </div>
  );
}
