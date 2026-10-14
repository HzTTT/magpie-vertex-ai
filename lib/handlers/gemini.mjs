import { modelUrl } from "../regions.mjs"
import { regionsFor, withRegions } from "../failover.mjs"
import { errorResponse } from "../errors.mjs"
import { saveMedia } from "../media.mjs"

const JSON_HEADERS = { "content-type": "application/json" }

export async function handle(req, ctx) {
  const body = requestBody(req)
  const token = await ctx.token()
  const res = await withRegions(
    regionsFor(req.entry, ctx, req.pinned),
    (loc) =>
      ctx.fetch(modelUrl(ctx.project, loc, "google", req.entry.id, "streamGenerateContent") + "?alt=sse", {
        method: "POST",
        headers: { "content-type": "application/json", authorization: `Bearer ${token}` },
        body: JSON.stringify(body),
        signal: req.signal,
      }),
    { signal: req.signal },
  )
  if (!res.ok) return geminiError(res)
  const stream = res.body.pipeThrough(new TextDecoderStream()).pipeThrough(lineRewriter(ctx)).pipeThrough(new TextEncoderStream())
  return new Response(stream, { status: res.status, headers: passHeaders(res.headers) })
}

// Vertex takes the model in the URL and rejects AI Studio cache ids (cachedContents/...).
function requestBody(req) {
  const body = { ...req.body }
  delete body.model
  if (typeof body.cachedContent === "string" && !body.cachedContent.startsWith("projects/")) delete body.cachedContent
  if (req.entry.family === "gemini-image") imageRequest(body)
  return body
}

// Image models reject function calling and thinking config, and must return text and images.
function imageRequest(body) {
  delete body.tools
  delete body.toolConfig
  const generationConfig = { ...body.generationConfig }
  delete generationConfig.thinkingConfig
  if (!generationConfig.responseModalities?.length) generationConfig.responseModalities = ["TEXT", "IMAGE"]
  body.generationConfig = generationConfig
}

// Line-based SSE rewrite: only `data:` lines that carry generated media change; every other line is re-emitted as received.
function lineRewriter(ctx) {
  let pending = ""
  return new TransformStream({
    async transform(text, controller) {
      const lines = (pending + text).split("\n")
      pending = lines.pop()
      for (const line of lines) controller.enqueue(await rewriteLine(line + "\n", ctx))
    },
    async flush(controller) {
      if (pending) controller.enqueue(await rewriteLine(pending, ctx))
    },
  })
}

async function rewriteLine(line, ctx) {
  const eol = line.endsWith("\n") ? "\n" : ""
  const text = line.slice(0, line.length - eol.length)
  const prefix = /^data: ?/.exec(text)
  if (!prefix) return line
  const payload = text.slice(prefix[0].length)
  if (!payload.includes('"inlineData"')) return line
  let chunk
  try {
    chunk = JSON.parse(payload)
  } catch {
    return line
  }
  if (!(await addMedia(chunk, ctx))) return line
  const trailing = payload.match(/\s*$/)[0]
  return `${prefix[0]}${JSON.stringify(chunk)}${trailing}${eol}`
}

// Saves each generated media part and adds a text part linking it right after the inlineData part.
async function addMedia(chunk, ctx) {
  let changed = false
  for (const candidate of chunk?.candidates ?? []) {
    const parts = candidate?.content?.parts
    if (!Array.isArray(parts)) continue
    const out = []
    for (const part of parts) {
      out.push(part)
      const kind = mediaKind(part?.inlineData?.mimeType)
      if (!kind) continue
      let saved
      try {
        saved = await saveMedia(ctx, { data: part.inlineData.data, mime: part.inlineData.mimeType })
      } catch (err) {
        ctx.log(`vertex: saving ${part.inlineData.mimeType} failed: ${err.message}`)
        continue
      }
      out.push({ text: mediaNote(kind, saved) })
      changed = true
    }
    candidate.content.parts = out
  }
  return changed
}

function mediaKind(mime) {
  const type = typeof mime === "string" ? mime.toLowerCase().split("/")[0] : ""
  return type === "image" || type === "audio" || type === "video" ? type : null
}

function mediaNote(kind, { path, url }) {
  const link = kind === "image" ? `![image](${url})` : `[${kind}](${url})`
  return `\n${link}\n\nSaved to: ${path}\n`
}

async function geminiError(res) {
  const text = await res.text()
  const parsed = parseJson(text)
  if (typeof parsed?.error?.message === "string") return new Response(JSON.stringify(parsed), { status: res.status, headers: JSON_HEADERS })
  return errorResponse("gemini", res.status, text || `HTTP ${res.status}`)
}

// Streaming methods wrap their error in an array: [{"error": {…}}].
function parseJson(text) {
  try {
    const value = JSON.parse(text)
    return Array.isArray(value) ? value[0] : value
  } catch {
    return undefined
  }
}

function passHeaders(headers) {
  const out = new Headers(headers)
  out.delete("content-length")
  out.delete("content-encoding")
  return out
}
