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

// ── Types ───────────────────────────────────────────────────────────────────

interface TaskStatus {
  status: "processing" | "completed" | "failed";
  progress: string;
  result?: {
    total_matched_jobs: number;
    sources: { adzuna: number; reed: number };
    total_recommendations: number;
    recommendations: Record<string, any>[];
    message?: string;
  };
  error?: string;
}

interface MatchResult {
  total_matched_jobs: number;
  sources: { adzuna: number; reed: number };
  total_recommendations: number;
  recommendations: Record<string, any>[];
  message?: string;
}

// ── API ─────────────────────────────────────────────────────────────────────

export const api = {
  health: () => request<{ status: string }>("/health"),

  scrape: () =>
    request<{
      total_jobs: number;
      sources: { adzuna: number; reed: number };
      jobs: Record<string, string>[];
    }>("/scrape"),

  /**
   * Upload a CV and poll for results.
   * @param onProgress - optional callback invoked with progress messages while processing
   */
  matchJobs: async (
    file: File,
    topN = 20,
    onProgress?: (progress: string) => void,
  ): Promise<MatchResult> => {
    // 1. Submit CV — returns immediately with task_id
    const form = new FormData();
    form.append("cv", file);
    form.append("top_n", String(topN));

    const submitRes = await fetch(`${BASE}/match-jobs`, {
      method: "POST",
      body: form,
    });

    if (!submitRes.ok) {
      const body = await submitRes.json().catch(() => ({}));
      throw new Error(body.detail || submitRes.statusText);
    }

    const { task_id } = (await submitRes.json()) as { task_id: string };

    // 2. Poll for results every 1.5s
    const POLL_INTERVAL = 1500;
    const MAX_POLLS = 120; // 3 minutes max

    for (let i = 0; i < MAX_POLLS; i++) {
      await new Promise((r) => setTimeout(r, POLL_INTERVAL));

      const pollRes = await fetch(`${BASE}/match-jobs/${task_id}`);
      if (!pollRes.ok) {
        throw new Error("Failed to check task status");
      }

      const status: TaskStatus = await pollRes.json();

      if (onProgress && status.progress) {
        onProgress(status.progress);
      }

      if (status.status === "completed" && status.result) {
        return status.result;
      }

      if (status.status === "failed") {
        throw new Error(status.error || "Job matching failed");
      }
    }

    throw new Error("Task timed out. Please try again.");
  },
};
