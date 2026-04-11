const BASE = "/api";

async function request<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...opts?.headers },
    ...opts,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || body.message || res.statusText);
  }
  return res.json();
}

// ── Health ───────────────────────────────────────────────────────────────────
export const api = {
  health: () => request<{ status: string }>("/health"),

  // ── Jobs ─────────────────────────────────────────────────────────────────
  scrape: () =>
    request<{
      total_jobs: number;
      sources: { adzuna: number; reed: number };
      jobs: Record<string, string>[];
    }>("/scrape"),

  // ── CV + Matching ────────────────────────────────────────────────────────
  matchJobs: (file: File, topN = 20) => {
    const form = new FormData();
    form.append("cv", file);
    form.append("top_n", String(topN));
    return fetch(`${BASE}/match-jobs`, { method: "POST", body: form }).then(
      async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body.detail || res.statusText);
        }
        return res.json() as Promise<{
          total_matched_jobs: number;
          sources: { adzuna: number; reed: number };
          total_recommendations: number;
          recommendations: Record<string, any>[];
          message?: string;
        }>;
      }
    );
  },
};
