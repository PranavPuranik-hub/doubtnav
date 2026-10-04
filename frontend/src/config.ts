export const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
export const WS_BASE = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws';

export const API_ENDPOINTS = {
  clips: `${API_BASE}/clips`,
  session: `${API_BASE}/session`,
  spawnObstacle: `${API_BASE}/sim/spawn_obstacle`,
  goal: `${API_BASE}/sim/goal`,
  reset: `${API_BASE}/sim/reset`,
  benchmark: `${API_BASE}/benchmark`,
};
