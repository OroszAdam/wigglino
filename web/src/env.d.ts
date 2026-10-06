/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL of the API, e.g. https://<user>-<space>.hf.space (no trailing slash). */
  readonly VITE_API_URL?: string
  /** Cloudflare Turnstile site key. Leave empty to disable the widget (local dev). */
  readonly VITE_TURNSTILE_SITE_KEY?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
