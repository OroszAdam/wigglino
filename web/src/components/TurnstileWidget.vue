<script setup lang="ts">
import { onBeforeUnmount, onMounted, useTemplateRef } from 'vue'

interface TurnstileApi {
  render(el: HTMLElement, opts: Record<string, unknown>): string
  reset(id: string): void
  remove(id: string): void
}
declare global {
  interface Window {
    turnstile?: TurnstileApi
  }
}

const SCRIPT_SRC = 'https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit'

const props = defineProps<{ siteKey: string }>()
const emit = defineEmits<{ token: [value: string] }>()
const el = useTemplateRef<HTMLDivElement>('el')
let widgetId: string | null = null

function loadScript(): Promise<TurnstileApi> {
  if (window.turnstile) return Promise.resolve(window.turnstile)
  return new Promise((resolve, reject) => {
    let script = document.querySelector<HTMLScriptElement>(`script[src="${SCRIPT_SRC}"]`)
    if (!script) {
      script = document.createElement('script')
      script.src = SCRIPT_SRC
      script.async = true
      document.head.appendChild(script)
    }
    script.addEventListener('load', () => (window.turnstile ? resolve(window.turnstile) : reject()))
    script.addEventListener('error', () => reject(new Error('Turnstile failed to load')))
  })
}

onMounted(async () => {
  const ts = await loadScript()
  if (!el.value) return
  widgetId = ts.render(el.value, {
    sitekey: props.siteKey,
    theme: 'dark',
    callback: (token: string) => emit('token', token),
    'expired-callback': () => emit('token', ''),
    'error-callback': () => emit('token', ''),
  })
})

onBeforeUnmount(() => {
  if (widgetId) window.turnstile?.remove(widgetId)
})

/** Tokens are single-use: call after each upload to get a fresh one. */
function reset() {
  emit('token', '')
  if (widgetId) window.turnstile?.reset(widgetId)
}

defineExpose({ reset })
</script>

<template>
  <div ref="el" class="min-h-[65px]" />
</template>
