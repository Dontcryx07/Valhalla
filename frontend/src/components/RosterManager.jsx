import { useEffect, useState } from "react";
import { useAuth } from "../hooks/useAuth";
import { apiUrl } from "../utils/api";

const panelStyle = {
  position: "fixed", right: 16, bottom: 64, width: 330, zIndex: 1100,
  border: "1px solid rgba(212,160,74,.32)", borderRadius: 6,
  background: "rgba(8,8,8,.96)", boxShadow: "0 14px 45px rgba(0,0,0,.62)",
  color: "#d0d0da", fontFamily: "'Outfit', sans-serif", padding: 13,
};
const controlStyle = { width: "100%", marginTop: 7, padding: "7px 8px", borderRadius: 3, border: "1px solid rgba(212,160,74,.28)", background: "#101010", color: "#ddd", fontFamily: "inherit", fontSize: 12 };

export default function RosterManager({ open, onClose, simulationRunning, onError, isAuthenticated }) {
  const { token } = useAuth();
  const [roster, setRoster] = useState([]);
  const [description, setDescription] = useState("");
  const [selected, setSelected] = useState("");
  const [newName, setNewName] = useState("");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");

  const refresh = async () => {
    try {
      const response = await fetch(apiUrl("/api/roster"), {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
      const data = await response.json();
      setRoster(data.agents || []);
      setSelected((previous) => previous || data.agents?.[0]?.id || "");
    } catch (error) { onError(error.message); }
  };
  useEffect(() => { if (open) refresh(); }, [open]);
  if (!open || !isAuthenticated) return null;

  const submit = async (path, payload, success) => {
    if (simulationRunning) { onError("Stop the simulation before editing the roster."); return; }
    setBusy(true); setNotice("");
    try {
      const response = await fetch(apiUrl(path), { method: "POST", headers: { "Content-Type": "application/json", ...(token ? { Authorization: `Bearer ${token}` } : {}) }, body: JSON.stringify(payload) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Roster change failed.");
      setNotice(`${success} Checkpoint ${data.checkpoint_tick} is now the latest timeline state.`);
      setDescription(""); setNewName("");
      await refresh();
    } catch (error) { onError(error.message); }
    finally { setBusy(false); }
  };

  return <aside style={panelStyle} aria-label="Simulation roster management">
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
      <strong style={{ color: "#e7bd70", fontSize: 12, letterSpacing: ".08em", textTransform: "uppercase" }}>Roster control</strong>
      <button onClick={onClose} style={{ color: "#aaa", background: "none", border: 0, cursor: "pointer", fontSize: 17 }}>×</button>
    </div>
    <p style={{ color: simulationRunning ? "#ff9494" : "#80df94", fontSize: 11, lineHeight: 1.35 }}>
      {simulationRunning ? "Stop the sim to edit membership." : "Stopped — edits create a new checkpoint; old checkpoints retain their original roster."}
    </p>
    <label style={{ display: "block", color: "#9898a4", fontSize: 10, marginTop: 12, letterSpacing: ".06em" }}>ADD AGENT — OBSERVER NOTES</label>
    <textarea value={description} onChange={(e) => setDescription(e.target.value)} rows={4} placeholder="Describe an adult student: name, age, branch, personality, interests, routine..." style={{ ...controlStyle, resize: "vertical" }} />
    <button disabled={busy || description.trim().length < 30} onClick={() => submit("/api/roster/add", { description }, "Agent added.")} style={{ ...controlStyle, color: "#80df94", cursor: "pointer" }}>GENERATE & ADD</button>
    <label style={{ display: "block", color: "#9898a4", fontSize: 10, marginTop: 15, letterSpacing: ".06em" }}>EXISTING AGENT</label>
    <select value={selected} onChange={(e) => setSelected(e.target.value)} style={controlStyle}>{roster.map((agent) => <option key={agent.id} value={agent.id}>{agent.name} · {agent.branch}</option>)}</select>
    <input value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="Replacement display name" style={controlStyle} />
    <button disabled={busy || !selected || newName.trim().length < 2} onClick={() => submit("/api/roster/replace", { agent_id: selected, name: newName.trim() }, "Name replaced.")} style={{ ...controlStyle, color: "#e7bd70", cursor: "pointer" }}>REPLACE NAME</button>
    <button disabled={busy || !selected} onClick={() => {
      const agent = roster.find((item) => item.id === selected);
      if (window.confirm(`Retire ${agent?.name || selected}? Their files will be archived, not deleted.`)) submit("/api/roster/remove", { agent_id: selected }, "Agent retired to the archive.");
    }} style={{ ...controlStyle, color: "#ff9494", cursor: "pointer" }}>REMOVE & ARCHIVE</button>
    {notice && <p style={{ marginTop: 10, color: "#80df94", fontSize: 10, lineHeight: 1.35 }}>{notice}</p>}
  </aside>;
}
