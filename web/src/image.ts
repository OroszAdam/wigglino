/** Longest side accepted by the API (MAX_SIDE); larger images would be downscaled there anyway. */
export const MAX_SIDE = 1280

/**
 * Downscale big photos before upload: faster on mobile and keeps under the size limit.
 * createImageBitmap applies EXIF orientation, so the result is upright.
 */
export async function prepareUpload(file: File): Promise<Blob> {
  const bitmap = await createImageBitmap(file)
  try {
    const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height))
    if (scale === 1 && file.size < 4 * 1024 * 1024 && /^image\/(jpeg|png|webp)$/.test(file.type)) {
      return file
    }
    const canvas = document.createElement('canvas')
    canvas.width = Math.round(bitmap.width * scale)
    canvas.height = Math.round(bitmap.height * scale)
    canvas.getContext('2d')!.drawImage(bitmap, 0, 0, canvas.width, canvas.height)
    return await new Promise<Blob>((resolve, reject) =>
      canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('encode failed'))), 'image/jpeg', 0.92),
    )
  } finally {
    bitmap.close()
  }
}
