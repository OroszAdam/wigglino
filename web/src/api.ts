export const API_URL = (import.meta.env.VITE_API_URL ?? 'http://localhost:7860').replace(/\/+$/, '')

export type DepthModel = 'mono' | 'metric'
export type Axis = 'vertical' | 'horizontal' | 'circular'

export interface RenderParams {
  pivot_x: number
  pivot_y: number
  frames: number
  rotation_deg: number
  depth_pct: number
  axis: Axis
  tear: number
  hole_fill: 'stretch' | 'push_pull' | 'none'
  order: 'ping_pong' | 'forward'
  fps: number
  format: 'gif' | 'webp'
}

export interface Upload {
  image_id: string
  width: number
  height: number
}

export interface JobStatus {
  status: 'queued' | 'running' | 'done' | 'error'
  stage: 'depth' | 'render' | 'encode' | null
  position: number | null
  error: string | null
  result: { url: string; depth_url: string; pivot_disparity: number; width: number; height: number } | null
}

export class ApiError extends Error {
  readonly status: number
  readonly retryAfter?: number

  constructor(message: string, status: number, retryAfter?: number) {
    super(message)
    this.status = status
    this.retryAfter = retryAfter
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${API_URL}${path}`, init)
  } catch {
    throw new ApiError('Cannot reach the server. It may be waking up, try again in a minute.', 0)
  }
  if (res.ok) return res.json() as Promise<T>
  let message = `Request failed (${res.status}).`
  try {
    const body = await res.json()
    if (typeof body.detail === 'string') message = body.detail
    else if (Array.isArray(body.detail)) message = 'Invalid settings.'
  } catch {
    /* non-JSON error body */
  }
  const retry = Number(res.headers.get('retry-after'))
  throw new ApiError(message, res.status, Number.isFinite(retry) && retry > 0 ? retry : undefined)
}

export function fileUrl(path: string, download = false): string {
  return `${API_URL}${path}${download ? '?download=1' : ''}`
}

export function uploadImage(file: Blob, turnstileToken: string): Promise<Upload> {
  const form = new FormData()
  form.append('file', file, 'upload.jpg')
  form.append('turnstile', turnstileToken)
  return request('/api/images', { method: 'POST', body: form })
}

export function createJob(imageId: string, model: DepthModel, params: RenderParams) {
  return request<{ job_id: string; position: number | null }>('/api/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_id: imageId, model, params }),
  })
}

export function getJob(jobId: string): Promise<JobStatus> {
  return request(`/api/jobs/${encodeURIComponent(jobId)}`)
}
