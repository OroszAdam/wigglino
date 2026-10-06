<script setup lang="ts">
import { computed, onBeforeUnmount, reactive, ref, useTemplateRef, watch } from 'vue'
import { ApiError, createJob, fileUrl, getJob, uploadImage } from './api'
import type { DepthModel, JobStatus, RenderParams } from './api'
import Aurora from './components/bits/Aurora.vue'
import GradientText from './components/bits/GradientText.vue'
import SpotlightCard from './components/bits/SpotlightCard.vue'
import StarBorder from './components/bits/StarBorder.vue'
import SegmentedControl from './components/SegmentedControl.vue'
import SliderField from './components/SliderField.vue'
import TurnstileWidget from './components/TurnstileWidget.vue'
import { prepareUpload } from './image'

const SITE_KEY = import.meta.env.VITE_TURNSTILE_SITE_KEY ?? ''
const MAX_FILE_MB = 30

const file = ref<File | null>(null)
const previewUrl = ref('')
const imageId = ref<string | null>(null)
const model = ref<DepthModel>('metric')
const params = reactive<RenderParams>({
  pivot_x: 0.5,
  pivot_y: 0.5,
  frames: 8,
  rotation_deg: 4.5,
  depth_pct: 30,
  axis: 'vertical',
  tear: 0.89,
  hole_fill: 'stretch',
  order: 'ping_pong',
  fps: 24,
  format: 'gif',
})

const token = ref('')
const turnstile = useTemplateRef<InstanceType<typeof TurnstileWidget>>('turnstile')
const busy = ref(false)
const status = ref('')
const error = ref('')
const elapsed = ref(0)
const result = ref<JobStatus['result']>(null)
const view = ref<'animation' | 'depth' | 'original'>('original')
const dragging = ref(false)

// Circular motion needs forward order and enough frames to look smooth.
watch(
  () => params.axis,
  (axis) => {
    params.order = axis === 'circular' ? 'forward' : 'ping_pong'
    if (axis === 'circular' && params.frames < 8) params.frames = 8
  },
)

const canGenerate = computed(() => !!file.value && !busy.value && (!!imageId.value || !SITE_KEY || !!token.value))
const shownUrl = computed(() => {
  if (view.value === 'animation' && result.value) return fileUrl(result.value.url)
  if (view.value === 'depth' && result.value) return fileUrl(result.value.depth_url)
  return previewUrl.value
})

function setFile(f: File | undefined) {
  if (!f || busy.value) return
  error.value = ''
  if (!f.type.startsWith('image/')) {
    error.value = 'Please choose an image file.'
    return
  }
  if (f.size > MAX_FILE_MB * 1024 * 1024) {
    error.value = `That file is over ${MAX_FILE_MB} MB.`
    return
  }
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  file.value = f
  previewUrl.value = URL.createObjectURL(f)
  imageId.value = null
  result.value = null
  view.value = 'original'
  params.pivot_x = 0.5
  params.pivot_y = 0.5
}

function onDrop(e: DragEvent) {
  dragging.value = false
  setFile(e.dataTransfer?.files[0])
}

function setPivot(e: MouseEvent) {
  const img = e.currentTarget as HTMLImageElement
  const r = img.getBoundingClientRect()
  params.pivot_x = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width))
  params.pivot_y = Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

function describe(s: JobStatus): string {
  if (s.status === 'queued') return s.position ? `Waiting in line, ${s.position} ahead of you…` : 'Starting…'
  if (s.stage === 'depth') return `Estimating depth with ${model.value} model…`
  if (s.stage === 'encode') return 'Encoding animation…'
  return 'Rendering frames…'
}

async function upload(): Promise<string> {
  status.value = 'Uploading…'
  const blob = await prepareUpload(file.value!)
  try {
    const up = await uploadImage(blob, token.value)
    return up.image_id
  } finally {
    turnstile.value?.reset()
  }
}

async function generate() {
  if (!canGenerate.value) return
  busy.value = true
  error.value = ''
  elapsed.value = 0
  const started = Date.now()
  const timer = setInterval(() => (elapsed.value = Math.round((Date.now() - started) / 1000)), 1000)
  try {
    imageId.value ??= await upload()
    let job
    try {
      job = await createJob(imageId.value, model.value, { ...params })
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) {
        imageId.value = null
        throw new ApiError('The uploaded photo expired. Complete the check and press Generate again.', 404)
      }
      throw e
    }
    let networkErrors = 0
    for (;;) {
      let s: JobStatus
      try {
        s = await getJob(job.job_id)
        networkErrors = 0
      } catch (e) {
        if (e instanceof ApiError && e.status === 0 && ++networkErrors < 5) {
          await sleep(2000)
          continue
        }
        throw e
      }
      if (s.status === 'done' && s.result) {
        result.value = s.result
        view.value = 'animation'
        break
      }
      if (s.status === 'error') throw new ApiError(s.error ?? 'Processing failed.', 500)
      status.value = describe(s)
      await sleep(1000)
    }
  } catch (e) {
    const err = e instanceof ApiError ? e : new ApiError('Something went wrong.', 0)
    error.value = err.retryAfter ? `${err.message} (retry in ${Math.ceil(err.retryAfter / 60)} min)` : err.message
  } finally {
    clearInterval(timer)
    busy.value = false
    status.value = ''
  }
}

onBeforeUnmount(() => {
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
})
</script>

<template>
  <div class="fixed inset-0 -z-10 opacity-60" aria-hidden="true">
    <Aurora :color-stops="['#5227FF', '#FF9FFC', '#5227FF']" :amplitude="1" :blend="0.5" />
  </div>

  <main class="mx-auto flex min-h-dvh max-w-6xl flex-col gap-8 px-4 py-10">
    <header class="space-y-3 text-center">
      <GradientText
        class-name="text-5xl font-bold tracking-tight sm:text-6xl"
        :colors="['#a78bfa', '#f0abfc', '#7dd3fc', '#a78bfa']"
        :animation-speed="6"
        >Wigglegram</GradientText
      >
      <p class="text-white/60">Turn a single photo into a wiggling 3D animation.</p>
    </header>

    <div class="grid gap-6 lg:grid-cols-[360px_1fr]">
      <SpotlightCard class-name="border-white/10 bg-black/40 backdrop-blur-xl !p-5" spotlight-color="rgba(167, 139, 250, 0.18)">
        <form class="space-y-5" @submit.prevent="generate">
          <label
            class="flex cursor-pointer flex-col items-center justify-center gap-1 rounded-2xl border border-dashed px-4 py-6 text-center text-sm transition-colors"
            :class="dragging ? 'border-violet-400 bg-violet-500/10' : 'border-white/20 hover:border-white/40'"
            @dragover.prevent="dragging = true"
            @dragleave="dragging = false"
            @drop.prevent="onDrop"
          >
            <input
              type="file"
              accept="image/jpeg,image/png,image/webp"
              class="sr-only"
              :disabled="busy"
              @change="setFile(($event.target as HTMLInputElement).files?.[0])"
            />
            <span class="font-medium text-white">{{ file ? file.name : 'Drop a photo or click to choose' }}</span>
            <span class="text-white/40">JPEG, PNG or WebP. Photos with a clear subject work best.</span>
          </label>

          <SegmentedControl
            v-model="model"
            label="Depth model"
            :disabled="busy"
            :options="[
              { value: 'metric', label: 'Metric', title: 'Depth Anything 3 Metric-Large' },
              { value: 'mono', label: 'Mono', title: 'Depth Anything 3 Mono-Large' },
            ]"
          />
          <SegmentedControl
            v-model="params.axis"
            label="Motion"
            :disabled="busy"
            :options="[
              { value: 'vertical', label: 'Side to side' },
              { value: 'horizontal', label: 'Up & down' },
              { value: 'circular', label: 'Circle' },
            ]"
          />
          <SliderField v-model="params.rotation_deg" label="Swing" :min="0.5" :max="15" :step="0.5" unit="°" />
          <SliderField v-model="params.depth_pct" label="Depth" :min="5" :max="100" :step="1" unit="%" />
          <SliderField v-model="params.frames" label="Frames" :min="params.axis === 'circular' ? 8 : 2" :max="16" :step="1" />
          <SliderField v-model="params.fps" label="Speed" :min="4" :max="30" :step="1" unit=" fps" />

          <details class="group space-y-4">
            <summary class="cursor-pointer text-xs font-medium uppercase tracking-wider text-white/50 hover:text-white/80">
              Advanced
            </summary>
            <div class="space-y-4 pt-2">
              <SliderField
                v-model="params.tear"
                label="Edge tearing"
                :min="0"
                :max="1"
                :step="0.01"
                hint="Lower values split foreground from background at depth edges more often."
              />
              <SegmentedControl
                v-model="params.hole_fill"
                label="Fill gaps"
                :options="[
                  { value: 'stretch', label: 'Stretch' },
                  { value: 'push_pull', label: 'Blur' },
                  { value: 'none', label: 'None' },
                ]"
              />
              <SegmentedControl
                v-model="params.order"
                label="Loop"
                :options="[
                  { value: 'ping_pong', label: 'Back & forth' },
                  { value: 'forward', label: 'Forward' },
                ]"
              />
              <SegmentedControl
                v-model="params.format"
                label="Format"
                :options="[
                  { value: 'gif', label: 'GIF' },
                  { value: 'webp', label: 'WebP' },
                ]"
              />
            </div>
          </details>

          <TurnstileWidget v-if="SITE_KEY" ref="turnstile" :site-key="SITE_KEY" @token="token = $event" />

          <StarBorder
            as="button"
            type="submit"
            :disabled="!canGenerate"
            color="#c4b5fd"
            speed="5s"
            :custom-class="['w-full', canGenerate ? '' : 'opacity-50 cursor-not-allowed'].join(' ')"
          >
            {{ busy ? `Working… ${elapsed}s` : result ? 'Generate again' : 'Generate' }}
          </StarBorder>

          <p v-if="status" class="text-center text-sm text-white/70" aria-live="polite">{{ status }}</p>
          <p v-if="error" class="rounded-xl bg-red-500/15 px-3 py-2 text-sm text-red-200" role="alert">{{ error }}</p>
        </form>
      </SpotlightCard>

      <SpotlightCard class-name="border-white/10 bg-black/40 backdrop-blur-xl !p-5 flex flex-col gap-4" spotlight-color="rgba(125, 211, 252, 0.12)">
        <div v-if="previewUrl" class="flex flex-wrap items-center justify-between gap-3">
          <SegmentedControl
            v-model="view"
            label="Show"
            class="min-w-64"
            :options="[
              { value: 'original', label: 'Photo' },
              ...(result
                ? [
                    { value: 'animation' as const, label: 'Wigglegram' },
                    { value: 'depth' as const, label: 'Depth' },
                  ]
                : []),
            ]"
          />
          <a
            v-if="result"
            :href="fileUrl(result.url, true)"
            class="rounded-xl border border-white/15 px-4 py-2 text-sm text-white hover:bg-white/10"
            >Download</a
          >
        </div>

        <div class="flex min-h-80 flex-1 items-center justify-center">
          <div v-if="previewUrl" class="relative inline-block">
            <img
              :src="shownUrl"
              alt="Preview"
              class="max-h-[70vh] w-auto cursor-crosshair rounded-2xl"
              draggable="false"
              @click="setPivot"
            />
            <span
              class="pointer-events-none absolute size-5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white shadow-[0_0_0_2px_rgba(0,0,0,0.5)]"
              :style="{ left: `${params.pivot_x * 100}%`, top: `${params.pivot_y * 100}%` }"
            />
          </div>
          <p v-else class="text-white/40">Your wigglegram will appear here.</p>
        </div>
        <p v-if="previewUrl" class="text-center text-xs text-white/40">
          Click the image to choose the focus point (the part that stays still).
        </p>
      </SpotlightCard>
    </div>

    <footer class="mt-auto space-y-1 text-center text-xs text-white/40">
      <p>Uploads and results are deleted automatically after about an hour.</p>
      <p>
        Depth by
        <a class="underline hover:text-white/70" href="https://github.com/ByteDance-Seed/Depth-Anything-3" rel="noopener" target="_blank"
          >Depth Anything 3</a
        >
        (Apache-2.0). UI components from
        <a class="underline hover:text-white/70" href="https://vue-bits.dev" rel="noopener" target="_blank">Vue Bits</a>.
      </p>
    </footer>
  </main>
</template>
