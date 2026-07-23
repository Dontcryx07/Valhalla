import { useState } from "react";

export default function InfoBar({ snapshot, showDebug, onToggleDebug, onFastForward, onSlowDown, onRewind, simulationRunning, onToggleSimulation, onToggleRoster }) {
  const [rewindAmount, setRewindAmount] = useState("10");
  const [rewindUnit, setRewindUnit] = useState("ticks");
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
      flexWrap: "wrap",
      gap: 14,
      rowGap: 6,
      maxWidth: "calc(100vw - 32px)",
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
      <span style={{ color: "#6b6b78" }}>|</span>
      <input
        type="number"
        min="1"
        max={rewindUnit === "ticks" ? 1440 : 24}
        value={rewindAmount}
        onChange={(event) => setRewindAmount(event.target.value)}
        aria-label="Rewind amount"
        style={{ pointerEvents: "auto", width: 42, border: "1px solid rgba(212,160,74,.3)", borderRadius: 3, background: "rgba(0,0,0,.45)", color: "#e7bd70", padding: "2px 4px", fontFamily: "'Space Mono', monospace", fontSize: 9 }}
      />
      <select
        value={rewindUnit}
        onChange={(event) => setRewindUnit(event.target.value)}
        aria-label="Rewind unit"
        style={{ pointerEvents: "auto", border: "1px solid rgba(212,160,74,.3)", borderRadius: 3, background: "rgba(0,0,0,.9)", color: "#e7bd70", padding: "2px", fontFamily: "'Space Mono', monospace", fontSize: 8 }}
      >
        <option value="ticks">ticks</option>
        <option value="hours">hours</option>
      </select>
      <button onClick={() => {
        const amount = Number(rewindAmount);
        if (!Number.isFinite(amount) || amount <= 0) return;
        onRewind(rewindUnit === "hours" ? { hours: amount } : { ticks: Math.round(amount) });
      }} style={{
        pointerEvents: "auto", border: "1px solid rgba(212,160,74,.3)", borderRadius: 3,
        background: "rgba(212,160,74,.08)", color: "#e7bd70", padding: "2px 5px",
        fontFamily: "'Space Mono', monospace", fontSize: 8, cursor: "pointer",
      }} title="Restore the requested number of simulated ticks or hours">
        REWIND
      </button>
      <button onClick={onFastForward} style={{
        pointerEvents: "auto", border: "1px solid rgba(81,207,102,.3)", borderRadius: 3,
        background: "rgba(81,207,102,.08)", color: "#80df94", padding: "2px 5px",
        fontFamily: "'Space Mono', monospace", fontSize: 8, cursor: "pointer",
      }} title="Double simulation speed">
        FAST FORWARD
      </button>
      <button onClick={onSlowDown} style={{
        pointerEvents: "auto", border: "1px solid rgba(91,155,213,.3)", borderRadius: 3,
        background: "rgba(91,155,213,.08)", color: "#8fbbe8", padding: "2px 5px",
        fontFamily: "'Space Mono', monospace", fontSize: 8, cursor: "pointer",
      }} title="Halve simulation speed (minimum 0.25x)">
        SLOW DOWN
      </button>
      <button onClick={onToggleSimulation} style={{
        pointerEvents: "auto", border: `1px solid ${simulationRunning ? "rgba(255,107,107,.42)" : "rgba(81,207,102,.42)"}`, borderRadius: 3,
        background: simulationRunning ? "rgba(255,107,107,.10)" : "rgba(81,207,102,.10)", color: simulationRunning ? "#ff9494" : "#80df94", padding: "2px 5px",
        fontFamily: "'Space Mono', monospace", fontSize: 8, cursor: "pointer",
      }} title={simulationRunning ? "Stop simulation ticks and retain checkpoints" : "Start from the newest available checkpoint"}>
        {simulationRunning ? "STOP SIM" : "START SIM"}
      </button>
      <button onClick={onToggleDebug} style={{
        pointerEvents: "auto", marginLeft: 2, border: "1px solid rgba(91,155,213,.3)", borderRadius: 3,
        background: showDebug ? "rgba(91,155,213,.22)" : "transparent", color: "#8fbbe8", padding: "2px 5px",
        fontFamily: "'Space Mono', monospace", fontSize: 8, cursor: "pointer",
      }}>
        {showDebug ? "DEBUG ON" : "DEBUG"}
      </button>
      <button onClick={onToggleRoster} style={{
        pointerEvents: "auto", border: "1px solid rgba(212,160,74,.3)", borderRadius: 3,
        background: "rgba(212,160,74,.08)", color: "#e7bd70", padding: "2px 5px",
        fontFamily: "'Space Mono', monospace", fontSize: 8, cursor: "pointer",
      }} title="Add, retire, or rename agents while the simulation is stopped">ROSTER</button>
    </div>
  );
}
