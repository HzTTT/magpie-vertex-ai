import {mkdir, writeFile} from "node:fs/promises"
import path from "node:path"
import {pathToFileURL} from "node:url"

const EXT = new Map([
  ["image/png", "png"],
  ["image/jpeg", "jpg"],
  ["image/webp", "webp"],
  ["image/gif", "gif"],
  ["audio/wav", "wav"],
  ["audio/x-wav", "wav"],
  ["audio/mpeg", "mp3"],
  ["audio/mp3", "mp3"],
  ["audio/ogg", "ogg"],
  ["audio/flac", "flac"],
  ["audio/l16", "pcm"],
  ["video/mp4", "mp4"],
  ["video/webm", "webm"],
  ["video/quicktime", "mov"],
])

export function extFor(mime = "") {
  const base = mime.split(";")[0].trim().toLowerCase()
  return EXT.get(base) ?? "bin"
}

// "audio/L16;codec=pcm;rate=24000" or "audio/l16; rate=24000; channels=1" -> {rate, channels}; anything else -> null
export function parsePcmMime(mime = "") {
  const [type, ...params] = mime.split(";").map((s) => s.trim())
  if (type.toLowerCase() !== "audio/l16") return null
  const opts = new Map(
    params.map((p) => {
      const i = p.indexOf("=")
      return i < 0 ? [p.toLowerCase(), ""] : [p.slice(0, i).trim().toLowerCase(), p.slice(i + 1).trim()]
    }),
  )
  if (opts.has("codec") && opts.get("codec").toLowerCase() !== "pcm") return null
  const rate = Number(opts.get("rate") ?? 24000)
  const channels = Number(opts.get("channels") ?? 1)
  if (!Number.isInteger(rate) || rate <= 0 || !Number.isInteger(channels) || channels <= 0) return null
  return {rate, channels}
}

// Canonical 44-byte RIFF/WAVE header in front of little-endian PCM samples.
export function pcmToWav(pcm, {rate = 24000, channels = 1, bits = 16} = {}) {
  const blockAlign = (channels * bits) / 8
  const out = new Uint8Array(44 + pcm.byteLength)
  const view = new DataView(out.buffer)
  const tag = (offset, text) => [...text].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)))
  tag(0, "RIFF")
  view.setUint32(4, 36 + pcm.byteLength, true)
  tag(8, "WAVE")
  tag(12, "fmt ")
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true) // PCM
  view.setUint16(22, channels, true)
  view.setUint32(24, rate, true)
  view.setUint32(28, rate * blockAlign, true)
  view.setUint16(32, blockAlign, true)
  view.setUint16(34, bits, true)
  tag(36, "data")
  view.setUint32(40, pcm.byteLength, true)
  out.set(pcm, 44)
  return out
}

export async function saveMedia(ctx, {data, mime, prefix = "vertex"}) {
  const bytes = typeof data === "string" ? Buffer.from(data, "base64") : data
  const dir = path.resolve(ctx.mediaDir)
  await mkdir(dir, {recursive: true})
  const file = path.join(dir, `${prefix}-${stamp()}-${randomHex(2)}.${extFor(mime)}`)
  await writeFile(file, bytes)
  return {path: file, url: pathToFileURL(file).href, bytes: bytes.byteLength}
}

// Local time, e.g. 20261014-093015
function stamp(d = new Date()) {
  const p = (n) => String(n).padStart(2, "0")
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`
}

function randomHex(bytes) {
  return [...crypto.getRandomValues(new Uint8Array(bytes))].map((b) => b.toString(16).padStart(2, "0")).join("")
}
