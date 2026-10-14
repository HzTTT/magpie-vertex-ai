import {modelUrl} from "../regions.mjs"
import {regionsFor, withRegions} from "../failover.mjs"
import {errorResponse} from "../errors.mjs"
import {saveMedia} from "../media.mjs"
import {lastUserTurn} from "../prompt.mjs"
import {chunk, geminiStream} from "../sse.mjs"

const DEFAULTS = {durationSeconds: 8, aspectRatio: "16:9", resolution: "720p", generateAudio: true}
const USAGE = {promptTokenCount: 0, candidatesTokenCount: 0, totalTokenCount: 0}

export async function handle(req, ctx) {
  const {text} = lastUserTurn(req.body)
  const frames = imageFrames(req.body)
  if (!text.trim()) return errorResponse("gemini", 400, "Veo needs a text prompt in the last user turn")
  const {entry, pinned} = req
  const pollMs = ctx.options.veoPollMs ?? 10_000
  const capMs = (ctx.options.veoTimeoutSec ?? 900) * 1000
  const body = {instances: [instance(text, frames, entry.id)], parameters: parameters(entry.id, ctx.options)}

  // Submit before the stream opens: a rejected job is an HTTP error, so magpie can fail over.
  let loc
  let submit
  try {
    submit = await withRegions(
      regionsFor(entry, ctx, pinned),
      (region) => {
        loc = region
        return post(ctx, modelUrl(ctx.project, region, "google", entry.id, "predictLongRunning"), body, req.signal)
      },
      {signal: req.signal},
    )
  } catch (err) {
    if (err.signIn) throw err
    return errorResponse("gemini", 502, `Veo submit unreachable: ${err.message}`)
  }
  if (!submit.ok) return failed(submit)
  const {name} = await submit.json()
  if (!name) return errorResponse("gemini", 502, "Veo submit returned no operation name")

  return geminiStream(async ({emit, keepalive, signal}) => {
    emit(chunk({text: `Veo job submitted (${entry.id}).`, thought: true}))

    // The operation lives in the region that accepted the submit, so polling never fails over.
    const started = Date.now()
    let op = {done: false}
    while (!op.done) {
      if (Date.now() - started >= capMs) throw new Error(`Veo did not finish within ${capMs / 1000}s`)
      await sleep(pollMs, signal)
      keepalive()
      const res = await post(ctx, modelUrl(ctx.project, loc, "google", entry.id, "fetchPredictOperation"), {operationName: name}, signal)
      if (!res.ok) throw new Error(`Veo poll failed (${res.status}): ${await res.text()}`)
      op = await res.json()
      if (!op.done) emit(chunk({text: `Veo generating… ${Math.round((Date.now() - started) / 1000)}s`, thought: true}))
    }

    if (op.error) throw new Error(`Veo failed: ${op.error.message ?? JSON.stringify(op.error)}`)
    // Some responses nest the payload under generateVideoResponse (the Gemini API shape).
    const result = op.response?.generateVideoResponse ?? op.response ?? {}
    const media = []
    for (const video of result.videos ?? []) {
      if (video.bytesBase64Encoded) {
        const saved = await saveMedia(ctx, {data: video.bytesBase64Encoded, mime: video.mimeType ?? "video/mp4", prefix: "veo"})
        media.push(`[video](${saved.url})`, `Path: ${saved.path}`)
      } else if (video.gcsUri) {
        media.push(`Stored at ${video.gcsUri} (not downloaded).`)
      }
    }
    const filtered = result.raiMediaFilteredCount ?? 0
    const notes = []
    if (filtered > 0) {
      const reasons = (result.raiMediaFilteredReasons ?? []).join("; ")
      notes.push(`${filtered} output(s) removed by Responsible AI filters${reasons ? `: ${reasons}` : "."}`)
    }
    if (!media.length && !filtered) throw new Error("Veo returned no video")
    emit(
      chunk({
        text: [media.length ? `Generated video with ${entry.id}.` : `${entry.id} produced no video.`, ...media, ...notes].join("\n"),
        finishReason: media.length ? "STOP" : "SAFETY",
        usage: USAGE,
      }),
    )
  }, {signal: req.signal})
}

// Upstream rejected the call before any stream opened: keep its status and Google's message.
async function failed(res) {
  const text = await res.text()
  let message = text
  try {
    message = JSON.parse(text).error?.message ?? text
  } catch {
    // Not JSON: the raw body is the message.
  }
  return errorResponse("gemini", res.status, message)
}

// lastUserTurn sorts media by mime, but first/last frame order is document order, so read images here.
function imageFrames(body) {
  const turn = (Array.isArray(body?.contents) ? body.contents : []).findLast((c) => c?.role === "user")
  return (turn?.parts ?? [])
    .filter((p) => p?.inlineData?.data && p.inlineData.mimeType?.startsWith("image/"))
    .map((p) => ({mime: p.inlineData.mimeType, data: p.inlineData.data}))
}

// Veo 2 takes neither resolution nor generateAudio. The second image is the last frame, Veo 3 only.
function instance(prompt, images, id) {
  const out = {prompt}
  if (images[0]) out.image = {bytesBase64Encoded: images[0].data, mimeType: images[0].mime}
  if (images[1] && id.startsWith("veo-3")) out.lastFrame = {bytesBase64Encoded: images[1].data, mimeType: images[1].mime}
  return out
}

function parameters(id, options = {}) {
  const opts = {...DEFAULTS, ...options.veo}
  const params = {
    sampleCount: 1,
    durationSeconds: opts.durationSeconds,
    aspectRatio: opts.aspectRatio,
    personGeneration: "allow_adult",
  }
  if (id.startsWith("veo-3")) Object.assign(params, {resolution: opts.resolution, generateAudio: opts.generateAudio})
  if (opts.negativePrompt) params.negativePrompt = opts.negativePrompt
  return params
}

function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal.aborted) return reject(signal.reason)
    const onAbort = () => {
      clearTimeout(timer)
      reject(signal.reason)
    }
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", onAbort)
      resolve()
    }, ms)
    signal.addEventListener("abort", onAbort, {once: true})
  })
}

async function post(ctx, url, body, signal) {
  const token = await ctx.token()
  return ctx.fetch(url, {
    method: "POST",
    headers: {authorization: `Bearer ${token}`, "content-type": "application/json"},
    body: JSON.stringify(body),
    signal,
  })
}
