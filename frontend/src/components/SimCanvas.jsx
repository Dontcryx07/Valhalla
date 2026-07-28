import { useEffect, useRef } from "react";
import { lerp } from "../utils/lerp";

const LERP_FACTOR = 0.12;
const CAM_LERP = 0.10;

// Night overlay opacity for a given sim hour (0..23): dark at night, clear midday.
function nightAlphaForHour(h) {
  // 0.85 at 00–04 and 22–24, ~0 between 09 and 17, smooth ramps between.
  if (h >= 22 || h < 5) return 0.82;
  if (h >= 9 && h <= 16) return 0.0;
  if (h >= 5 && h < 9) return 0.82 * (1 - (h - 5) / 4);   // dawn ramp down
  return 0.82 * ((h - 16) / 6);                            // dusk ramp up (17–22)
}

export default function SimCanvas({ snapshot, focusedId, onFocus }) {
  const canvasRef = useRef(null);
  const mapImgRef = useRef(null);
  const nightImgRef = useRef(null);
  const agentsRef = useRef({});
  const rafRef = useRef(null);
  const nightAlphaRef = useRef(0);
  const camRef = useRef({ x: 0, y: 0, tx: 0, ty: 0, userPan: false });
  const viewRef = useRef({ ox: 0, oy: 0, scale: 1 });
  const focusRef = useRef(null);

  useEffect(() => { focusRef.current = focusedId; }, [focusedId]);

  // Load map images once
  useEffect(() => {
    const img = new Image();
    img.src = "/map.png";
    img.onload = () => { mapImgRef.current = img; startLoop(); };
    const night = new Image();
    night.src = "/map_night.png";
    night.onload = () => { nightImgRef.current = night; };
  }, []);

  // Update targets from snapshot
  useEffect(() => {
    if (!snapshot?.agents) return;
    if (snapshot.time) {
      const h = parseInt(String(snapshot.time).split(":")[0], 10) || 0;
      nightAlphaRef.current = nightAlphaForHour(h);
    }
    for (const [id, data] of Object.entries(snapshot.agents)) {
      if (!agentsRef.current[id]) {
        agentsRef.current[id] = { x: data.position.x, y: data.position.y };
      }
      const a = agentsRef.current[id];
      a.tx = data.position.x;
      a.ty = data.position.y;
      a.color = data.color;
      a.name = data.name;
      a.activity = data.activity || "";
      a.inConversation = data.in_conversation;
    }
    if (!rafRef.current) startLoop();
  }, [snapshot]);

  function startLoop() {
    if (rafRef.current) return;
    function tick() {
      const canvas = canvasRef.current;
      if (!canvas) { rafRef.current = null; return; }
      const ctx = canvas.getContext("2d");
      const W = (canvas.width = window.innerWidth);
      const H = (canvas.height = window.innerHeight);
      const img = mapImgRef.current;

      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = "#000000";
      ctx.fillRect(0, 0, W, H);
      if (!img) { rafRef.current = requestAnimationFrame(tick); return; }

      const iw = img.width, ih = img.height;
      const scale = Math.min(W / iw, H / ih) * 0.92;
      const baseOx = (W - iw * scale) / 2;
      const baseOy = (H - ih * scale) / 2;

      // Camera: follow focused agent unless the user is panning.
      const cam = camRef.current;
      const fid = focusRef.current;
      if (fid && agentsRef.current[fid] && !cam.userPan) {
        const a = agentsRef.current[fid];
        cam.tx = W / 2 - (baseOx + a.x * scale);
        cam.ty = H / 2 - (baseOy + a.y * scale);
      }
      cam.x = lerp(cam.x, cam.tx, CAM_LERP);
      cam.y = lerp(cam.y, cam.ty, CAM_LERP);
      const ox = baseOx + cam.x, oy = baseOy + cam.y;
      viewRef.current = { ox, oy, scale };

      // advance dot positions
      for (const id of Object.keys(agentsRef.current)) {
        const a = agentsRef.current[id];
        if (a.tx == null) continue;
        a.x = lerp(a.x, a.tx, LERP_FACTOR);
        a.y = lerp(a.y, a.ty, LERP_FACTOR);
      }

      ctx.save();
      ctx.translate(ox, oy);
      ctx.scale(scale, scale);
      ctx.drawImage(img, 0, 0);
      // Day/night blend: overlay the night map at a time-based opacity.
      const na = nightAlphaRef.current;
      if (nightImgRef.current && na > 0.01) {
        ctx.globalAlpha = na;
        ctx.drawImage(nightImgRef.current, 0, 0);
        ctx.globalAlpha = 1;
      }

      // draw agents
      for (const id of Object.keys(agentsRef.current)) {
        const a = agentsRef.current[id];
        if (a.x == null) continue;
        const focused = id === fid;
        const r = focused ? 7 : 4;

        // glow / focus ring
        const glowR = focused ? 22 : 11;
        const glow = ctx.createRadialGradient(a.x, a.y, 0, a.x, a.y, glowR);
        glow.addColorStop(0, (a.color || "#fff") + (focused ? "90" : "50"));
        glow.addColorStop(1, (a.color || "#fff") + "00");
        ctx.beginPath();
        ctx.arc(a.x, a.y, glowR, 0, Math.PI * 2);
        ctx.fillStyle = glow;
        ctx.fill();

        if (a.inConversation) {
          ctx.beginPath();
          ctx.arc(a.x, a.y, r + 5, 0, Math.PI * 2);
          ctx.strokeStyle = "#d4a04a";
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }

        ctx.beginPath();
        ctx.arc(a.x, a.y, r, 0, Math.PI * 2);
        ctx.fillStyle = a.color || "#fff";
        ctx.fill();
        ctx.strokeStyle = focused ? "#fff" : "rgba(255,255,255,0.7)";
        ctx.lineWidth = focused ? 2 : 1.2;
        ctx.stroke();

        // labels (name + short activity)
        const fontPx = Math.max(9, 11 / scale);
        ctx.font = `${fontPx}px 'Outfit', sans-serif`;
        ctx.textAlign = "center";
        ctx.fillStyle = "#fff";
        ctx.strokeStyle = "rgba(0,0,0,0.7)";
        ctx.lineWidth = 3 / scale;
        const ly = a.y - r - 6 / scale;
        ctx.strokeText(a.name || id, a.x, ly);
        ctx.fillText(a.name || id, a.x, ly);
        if (focused && a.activity) {
          const af = Math.max(8, 9 / scale);
          ctx.font = `${af}px 'Space Mono', monospace`;
          ctx.fillStyle = "#d4a04a";
          const act = a.activity.length > 40 ? a.activity.slice(0, 40) + "…" : a.activity;
          ctx.strokeText(act, a.x, a.y + r + 12 / scale);
          ctx.fillText(act, a.x, a.y + r + 12 / scale);
        }
      }
      ctx.restore();

      // vignette
      const vig = ctx.createRadialGradient(W / 2, H / 2, H * 0.15, W / 2, H / 2, H * 0.78);
      vig.addColorStop(0, "rgba(0,0,0,0)");
      vig.addColorStop(1, "rgba(0,0,0,0.5)");
      ctx.fillStyle = vig;
      ctx.fillRect(0, 0, W, H);

      rafRef.current = requestAnimationFrame(tick);
    }
    rafRef.current = requestAnimationFrame(tick);
  }

  // Click-to-focus + drag-to-pan
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let dragging = false, moved = false, sx = 0, sy = 0, startCam = null;

    function onDown(e) {
      dragging = true; moved = false;
      sx = e.clientX; sy = e.clientY;
      startCam = { ...camRef.current };
    }
    function onMove(e) {
      if (!dragging) return;
      const dx = e.clientX - sx, dy = e.clientY - sy;
      if (Math.abs(dx) + Math.abs(dy) > 4) moved = true;
      camRef.current.userPan = true;
      camRef.current.tx = startCam.tx + dx;
      camRef.current.ty = startCam.ty + dy;
    }
    function onUp(e) {
      dragging = false;
      if (moved) return;
      // treat as a click: hit-test agents
      const { ox, oy, scale } = viewRef.current;
      const mx = (e.clientX - ox) / scale;
      const my = (e.clientY - oy) / scale;
      let best = null, bestD = 18 / scale;
      for (const id of Object.keys(agentsRef.current)) {
        const a = agentsRef.current[id];
        const d = Math.hypot(a.x - mx, a.y - my);
        // Names are rendered above the marker, so make the visible name area
        // clickable too. This keeps the map as the sole, uncluttered roster.
        const nameHalfWidth = Math.max(22, (a.name || id).length * 3.6) / scale;
        const inName = Math.abs(a.x - mx) <= nameHalfWidth
          && my >= a.y - 24 / scale
          && my <= a.y - 2 / scale;
        if (inName || d < bestD) { bestD = d; best = id; }
      }
      if (onFocus) {
        if (best) { camRef.current.userPan = false; onFocus(best); }
        else onFocus(null);
      }
    }
    canvas.addEventListener("mousedown", onDown);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    const onResize = () => { if (!rafRef.current) startLoop(); };
    window.addEventListener("resize", onResize);
    return () => {
      canvas.removeEventListener("mousedown", onDown);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      window.removeEventListener("resize", onResize);
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    };
  }, [onFocus]);

  return (
    <canvas
      ref={canvasRef}
      style={{ position: "fixed", top: 0, left: 0, cursor: "grab", display: "block" }}
    />
  );
}
