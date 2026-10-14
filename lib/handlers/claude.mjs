import { modelUrl } from "../regions.mjs"
import { regionsFor, withRegions } from "../failover.mjs"
import { errorResponse } from "../errors.mjs"

const VERTEX_VERSION = "vertex-2023-10-16"
const JSON_HEADERS = { "content-type": "application/json" }

export async function handle(req, ctx) {
  const body = requestBody(req)
  const token = await ctx.token()
  const method = req.stream ? "streamRawPredict" : "rawPredict"
  const res = await withRegions(
    regionsFor(req.entry, ctx, req.pinned),
    (loc) =>
      ctx.fetch(modelUrl(ctx.project, loc, "anthropic", req.entry.id, method), {
        method: "POST",
        headers: { "content-type": "application/json", authorization: `Bearer ${token}` },
        body: JSON.stringify(body),
        signal: req.signal,
      }),
    { signal: req.signal },
  )
  if (!res.ok) return anthropicError(res)
  return new Response(res.body, { status: res.status, headers: passHeaders(res.headers) })
}

// Vertex wants the model in the URL, the API version in the body, and betas as a body array instead of a header.
function requestBody(req) {
  const body = { ...req.body, anthropic_version: VERTEX_VERSION }
  delete body.model
  const betas = [...new Set([...betaList(body.anthropic_beta), ...betaList(req.headers.get("anthropic-beta"))])]
  if (betas.length) body.anthropic_beta = betas
  else delete body.anthropic_beta
  dropWebSearch(body)
  return body
}

function betaList(value) {
  return [value]
    .flat()
    .filter((v) => typeof v === "string")
    .flatMap((v) => v.split(","))
    .map((v) => v.trim())
    .filter(Boolean)
}

// Magpie injects its own web_search server tool, which Vertex does not offer. A tool_choice without tools is rejected.
function dropWebSearch(body) {
  if (!Array.isArray(body.tools)) return
  const tools = body.tools.filter((tool) => !String(tool?.type ?? "").startsWith("web_search_"))
  if (tools.length) {
    body.tools = tools
    return
  }
  delete body.tools
  delete body.tool_choice
}

async function anthropicError(res) {
  const text = await res.text()
  const parsed = parseJson(text)
  if (parsed?.type === "error") return new Response(text, { status: res.status, headers: JSON_HEADERS })
  const message = parsed?.error?.message
  return errorResponse("anthropic", res.status, typeof message === "string" ? message : text || `HTTP ${res.status}`)
}

// streamRawPredict wraps its error in an array: [{"error": {…}}].
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
