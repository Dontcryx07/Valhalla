import { useEffect, useRef, useState, useCallback } from "react";
import { wsUrl } from "../utils/api";

export default function useSimState() {
  const [snapshot, setSnapshot] = useState(null);
  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);
  const disposedRef = useRef(false);

  const connect = useCallback(() => {
    if (disposedRef.current) return;
    clearTimeout(reconnectTimer.current);
    const url = wsUrl("/ws/sim");
    const ws = new WebSocket(url);

    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        setSnapshot(data);
      } catch { /* ignore parse errors */ }
    };

    ws.onclose = () => {
      // Closing a socket during React cleanup is intentional. Reconnecting in
      // that case creates an orphan connection after the view has unmounted.
      if (disposedRef.current || wsRef.current !== ws) return;
      reconnectTimer.current = setTimeout(connect, 3000);
    };

    ws.onerror = () => ws.close();
    wsRef.current = ws;
  }, []);

  useEffect(() => {
    disposedRef.current = false;
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      disposedRef.current = true;
      const ws = wsRef.current;
      wsRef.current = null;
      if (ws) {
        ws.onclose = null;
        ws.close();
      }
    };
  }, [connect]);

  return snapshot;
}
