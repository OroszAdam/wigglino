<script setup lang="ts">
defineProps<{ label: string; min: number; max: number; step: number; unit?: string; hint?: string }>()
const model = defineModel<number>({ required: true })
</script>

<template>
  <label class="block space-y-1" :title="hint">
    <span class="flex justify-between text-xs font-medium uppercase tracking-wider text-white/50">
      <span>{{ label }}</span>
      <span class="tabular-nums text-white/80">{{ model }}{{ unit }}</span>
    </span>
    <input v-model.number="model" type="range" class="w-full" :min="min" :max="max" :step="step" />
  </label>
</template>
