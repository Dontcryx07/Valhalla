const BASE = import.meta.env.VITE_API_URL || "";

export function apiUrl(path) {
  return `${BASE}${path}`;
}

export function wsUrl(path) {
  if (BASE) {
    return `${BASE.replace(/^http/, "ws")}${path}`;
  }
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${location.host}${path}`;
}
