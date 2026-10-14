import { openapiUrl } from "../regions.mjs"
import { regionsFor, withRegions } from "../failover.mjs"
import { errorResponse } from "../errors.mjs"

const JSON_HEADERS = { "content-type": "application/json" }

export async function handle(req, ctx) {
  const body = requestBody(req)
  const token = await ctx.token()
  const res = await withRegions(
    regionsFor(req.entry, ctx, req.pinned),
    (loc) =>
      ctx.fetch(openapiUrl(ctx.project, loc, "/chat/completions"), {
        method: "POST",
        headers: { "content-type": "application/json", authorization: `Bearer ${token}` },
        body: JSON.stringify(body),
        signal: req.signal,
      }),
    { signal: req.signal },
  )
  if (!res.ok) return openaiError(res)
  return new Response(res.body, { status: res.status, headers: passHeaders(res.headers) })
}

// The OpenAI-compatible surface takes the publisher-qualified id. No other field is rewritten: a live probe
// was accepted with store, parallel_tool_calls, user and stream_options, so nothing is dropped.
function requestBody(req) {
  const body = { ...req.body, model: `${req.entry.publisher}/${req.entry.id}` }
  if (body.stream && !body.stream_options) body.stream_options = { include_usage: true }
  return body
}

async function openaiError(res) {
  const text = await res.text()
  const parsed = parseJson(text)
  if (typeof parsed?.error?.type === "string") return new Response(text, { status: res.status, headers: JSON_HEADERS })
  const message = parsed?.error?.message
  return errorResponse("openai", res.status, typeof message === "string" ? message : text || `HTTP ${res.status}`)
}

// Vertex wraps some errors in an array: [{"error": {…}}].
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
