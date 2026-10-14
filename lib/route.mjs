import { lookup } from "./catalog.mjs"
import { errorResponse } from "./errors.mjs"
import * as claude from "./handlers/claude.mjs"
import * as embeddings from "./handlers/embeddings.mjs"
import * as gemini from "./handlers/gemini.mjs"
import * as lyria from "./handlers/lyria.mjs"
import * as maas from "./handlers/maas.mjs"
import * as transcribe from "./handlers/transcribe.mjs"
import * as tts from "./handlers/tts.mjs"
import * as veo from "./handlers/veo.mjs"
import { BASE } from "./regions.mjs"

export const HANDLERS = { claude, embeddings, gemini, lyria, maas, transcribe, tts, veo }

// Endpoint → model family → handler. "/models/" covers every /models/<id>:<method> path.
const ENDPOINTS = {
  "/models/": { gemini: "gemini", "gemini-image": "gemini", "gemini-tts": "tts", "chirp-tts": "tts", veo: "veo", lyria: "lyria" },
  "/messages": { claude: "claude" },
  // Vertex's OpenAI-compatible endpoint also serves Gemini chat models (as google/<id>).
  "/chat/completions": { maas: "maas", gemini: "maas" },
  "/embeddings": { embedding: "embeddings" },
  "/audio/transcriptions": { gemini: "transcribe", "gemini-image": "transcribe" },
  "/audio/translations": { gemini: "transcribe", "gemini-image": "transcribe" },
}

// Where a family is served, for the 400 that names it.
const HOME = {
  gemini: "/models/<id>:streamGenerateContent",
  "gemini-image": "/models/<id>:streamGenerateContent",
  "gemini-tts": "/models/<id>:streamGenerateContent",
  "chirp-tts": "/models/<id>:streamGenerateContent",
  veo: "/models/<id>:predictLongRunning",
  lyria: "/models/<id>:predict",
  claude: "/messages",
  maas: "/chat/completions",
  embedding: "/embeddings",
}

// Key headers never reach a handler: handlers authenticate with ctx.token().
const KEY_HEADERS = ["authorization", "x-api-key", "x-goog-api-key"]
const V1 = new URL(BASE).pathname

export async function route(input, init, ctx, { entries, handlers = HANDLERS } = {}) {
  const url = new URL(typeof input === "string" ? input : (input.href ?? input.url))
  const path = (url.pathname.startsWith(V1) ? url.pathname.slice(V1.length) : url.pathname).replace(/%3A/gi, ":")
  const protocol = protocolOf(path)
  const headers = new Headers(init?.headers)
  for (const name of KEY_HEADERS) headers.delete(name)

  const parsed = await parseBody(url, headers, init?.body, protocol)
  if (parsed.error) return parsed.error
  const { body, form } = parsed

  const endpoint = path.startsWith("/models/") ? "/models/" : path
  const table = ENDPOINTS[endpoint]
  if (!table) return errorResponse(protocol, 404, `vertex-ai has no endpoint ${path}`)

  const apiId = modelOf(path, body, form)
  const found = apiId ? lookup(apiId, entries) : null
  if (!found) return errorResponse(protocol, 404, apiId ? `model ${apiId} is not served by vertex-ai` : "the request names no model")
  const { entry, pinned } = found

  const name = table[entry.family]
  if (!name) return errorResponse(protocol, 400, `${apiId} is a ${entry.family} model and does not take ${path}; send it to ${HOME[entry.family] ?? "its own endpoint"}`)

  const signal = init?.signal
  const req = {
    protocol,
    suffix: path + url.search,
    entry,
    pinned,
    apiId,
    body,
    raw: init?.body,
    headers,
    stream: path.includes("streamGenerateContent") || body?.stream === true,
    signal,
    form,
  }
  try {
    return await handlers[name].handle(req, ctx)
  } catch (e) {
    if (e?.signIn || e?.name === "AbortError" || signal?.aborted) throw e
    ctx?.log?.(`${apiId}: ${e?.message ?? e}`)
    return errorResponse(protocol, 502, `vertex-ai ${apiId}: ${e?.message ?? e}`)
  }
}

function protocolOf(path) {
  if (path.startsWith("/models/")) return "gemini"
  if (path === "/messages") return "anthropic"
  return "openai"
}

// The body as JSON, or as the parsed multipart form (req.body is null then).
async function parseBody(url, headers, raw, protocol) {
  if ((headers.get("content-type") ?? "").toLowerCase().startsWith("multipart/form-data")) {
    const h = new Headers(headers)
    h.delete("content-length")
    try {
      return { body: null, form: await new Request(url.href, { method: "POST", headers: h, body: raw }).formData() }
    } catch (e) {
      return { error: errorResponse(protocol, 400, `cannot read the multipart form: ${e?.message ?? e}`) }
    }
  }
  const text = raw == null ? "" : typeof raw === "string" ? raw : new TextDecoder().decode(raw)
  if (!text.trim()) return { body: null, form: null }
  try {
    return { body: JSON.parse(text), form: null }
  } catch {
    return { error: errorResponse(protocol, 400, "the request body is not valid JSON") }
  }
}

function modelOf(path, body, form) {
  const gem = /^\/models\/(.+):[A-Za-z]+$/.exec(path)
  if (gem) return safeDecode(gem[1])
  if (typeof body?.model === "string") return body.model
  const model = form?.get("model")
  return typeof model === "string" ? model : null
}

function safeDecode(s) {
  try {
    return decodeURIComponent(s)
  } catch {
    return s
  }
}
