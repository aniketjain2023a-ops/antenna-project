/** Typed client for the FastAPI backend (all routes under /api). */

export type Direction = 'minimize' | 'maximize' | null

export interface Feature {
  name: string
  short: string
  unit: string
  description: string
  group: string
  min: number
  max: number
  median: number
  optimize_default: boolean
}

export interface Target {
  name: string
  short: string
  unit: string
  description: string
  direction: Direction
  n_values: number
}

export interface Experiment {
  id: number
  target: string
  target_short: string
  unit: string
  model: string
  model_label: string
  variant: string
  r2: number | null
  mae: number | null
  rmse: number | null
  cv_r2_mean: number | null
  cv_r2_std: number | null
  n_train: number
  n_test: number
  created_at: string
  run_group: string
  active: boolean
  n_samples_used?: number
}

export interface Dataset {
  id: number
  key: string
  label: string
  name: string
  type: 'SYNTHETIC' | 'SIMULATION' | 'MEASUREMENT' | 'PUBLIC'
  type_description: string
  provenance: string
  description: string
  sample_count: number
  checksum: string
  created_at: string
  frequency_note: string
  features: Feature[]
  targets: Target[]
  quality: { rows: number; missing_values: number; duplicate_rows: number }
  active_models: Record<string, Experiment>
  ready: boolean
}

export interface UnavailableSource {
  key: string
  label: string
  message: string
  stages: Record<string, boolean>
}

export interface DatasetsResponse {
  datasets: Dataset[]
  unavailable: UnavailableSource[]
  models: { key: string; label: string; description: string }[]
  default_models: string[]
}

export interface Prediction {
  target: string
  short: string
  unit: string
  value: number | null
  uncertainty: number | null
  model: string
  model_label: string
  r2: number | null
  rmse: number | null
  partial_training: boolean
  n_samples_used: number | null
}

export interface PredictResponse {
  inputs: Record<string, number>
  predictions: Prediction[]
  extrapolation: boolean
  warnings: string[]
  saved_id: number | null
}

export interface Design {
  label: string
  parameters: Record<string, number>
  predictions: Record<string, number | null>
  uncertainty: Record<string, number | null>
  feasible: boolean
  violations: Record<string, number>
}

export interface OptimizeRequest {
  objective: string
  constraint: { target: string; operator: '<=' | '>='; limit: number } | null
  bounds: Record<string, [number, number]>
  baseline: Record<string, number>
  budget: number
  seed: number
}

export interface OptimizeResponse {
  run_id: number
  objective: string
  direction: Direction
  constraint: OptimizeRequest['constraint']
  free_parameters: string[]
  fixed_parameters: Record<string, number>
  baseline: Design
  candidate: Design
  history: { evaluation: number; best: number }[]
  n_evals: number
  runtime_s: number
  nearest_simulated: {
    design: string
    row: number
    distance: number
    values: { target: string; short: string; unit: string; dataset: number | null; predicted: number | null }[]
  }[] | null
  synthetic_oracle: { design: string; S11_dB: number; Gain_dBi: number }[] | null
}

export interface Diagnostics {
  experiment: Experiment
  points: { actual: number; predicted: number; residual: number }[]
  importance: {
    permutation: { feature: string; short: string; mean: number; std: number }[] | null
    impurity: { feature: string; short: string; mean: number }[] | null
  }
}

export interface ResultsSummary {
  metrics: (Experiment & { dataset_id: number; dataset_label: string; dataset_type: string })[]
  optimization_runs: {
    id: number
    created_at: string
    dataset_label: string
    dataset_id: number
    algorithm: string
    objective: string
    objective_short: string
    direction: Direction
    unit: string
    baseline: Record<string, number>
    candidate: Record<string, number>
    baseline_value: number | null
    candidate_value: number | null
    constrained: boolean
    feasible: boolean
    evaluations: number
  }[]
  counts: Record<string, number>
}

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`/api${path}`, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    })
  } catch {
    throw new ApiError(0, 'Cannot reach the MetaOpt-AI backend. Is it running on port 8000?')
  }
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* keep statusText */
    }
    throw new ApiError(res.status, detail)
  }
  return res.json() as Promise<T>
}

export const api = {
  datasets: () => request<DatasetsResponse>('/datasets'),
  preview: (id: number) => request<{ columns: string[]; rows: (number | null)[][] }>(`/datasets/${id}/preview`),
  experiments: (id: number) => request<{ experiments: Experiment[] }>(`/datasets/${id}/experiments`),
  train: (id: number, models: string[]) =>
    request<{ experiments: Experiment[]; seconds: number }>(`/datasets/${id}/train`, {
      method: 'POST',
      body: JSON.stringify({ models }),
    }),
  predict: (id: number, params: Record<string, number>, save = false) =>
    request<PredictResponse>(`/datasets/${id}/predict`, { method: 'POST', body: JSON.stringify({ params, save }) }),
  optimize: (id: number, body: OptimizeRequest) =>
    request<OptimizeResponse>(`/datasets/${id}/optimize`, { method: 'POST', body: JSON.stringify(body) }),
  diagnostics: (experimentId: number) => request<Diagnostics>(`/experiments/${experimentId}/diagnostics`),
  results: () => request<ResultsSummary>('/results/summary'),
  exportUrl: (table: 'datasets' | 'experiments' | 'predictions' | 'optimizations') => `/api/export/${table}.csv`,
}
