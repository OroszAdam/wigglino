<script setup lang="ts" generic="T extends string">
defineProps<{ options: { value: T; label: string; title?: string }[]; label: string; disabled?: boolean }>()
const model = defineModel<T>({ required: true })
</script>

<template>
  <fieldset class="space-y-1.5" :disabled="disabled">
    <legend class="text-xs font-medium uppercase tracking-wider text-white/50">{{ label }}</legend>
    <div class="grid auto-cols-fr grid-flow-col gap-1 rounded-xl border border-white/10 bg-black/30 p-1">
      <label
        v-for="o in options"
        :key="o.value"
        :title="o.title"
        class="cursor-pointer rounded-lg px-2 py-1.5 text-center text-sm transition-colors has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-violet-400"
        :class="model === o.value ? 'bg-violet-500/30 text-white' : 'text-white/60 hover:text-white'"
      >
        <input v-model="model" type="radio" class="sr-only" :value="o.value" />
        {{ o.label }}
      </label>
    </div>
  </fieldset>
</template>
