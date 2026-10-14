import { regionsFor, withRegions } from "../failover.mjs"
import { modelUrl } from "../regions.mjs"
import { errorResponse } from "../errors.mjs"

const CONCURRENCY = 4

// Per-family Vertex call. `batch` is inputs per call. `read(json, n)` returns
// [{values, tokens}] for the n inputs, or null when the reply has another shape.
const TEXT_PREDICT = {
  method: "predict",
  batch: 250,
  payload: (texts, { dimensions, taskType }) => ({
    instances: texts.map(content => ({ content, task_type: taskType })),
    parameters: { autoTruncate: true, outputDimensionality: dimensions },
  }),
  read: (json, n) => predictions(json, n, p => ({
    values: p.embeddings?.values,
    tokens: p.embeddings?.statistics?.token_count ?? 0,
  })),
}

const MULTIMODAL = {
  method: "predict",
  batch: 1,
  payload: (texts, { dimensions }) => ({
    instances: texts.map(text => ({ text })),
    parameters: dimensions === undefined ? undefined : { dimension: dimensions },
  }),
  read: (json, n) => predictions(json, n, p => ({ values: p.textEmbedding, tokens: 0 })),
}

// gemini-embedding-2 answers :embedContent (not :predict), one content per call.
const EMBED_CONTENT = {
  method: "embedContent",
  batch: 1,
  payload: ([text], { dimensions, taskType }) => ({
    content: { parts: [{ text }] },
    taskType,
    outputDimensionality: dimensions,
  }),
  read: json => {
    const values = json?.embedding?.values
    return Array.isArray(values) ? [{ values, tokens: json.usageMetadata?.promptTokenCount ?? 0 }] : null
  },
}

export async function handle(req, ctx) {
  const input = parseBody(req.body)
  if (input.error) return errorResponse("openai", 400, input.error)

  const mode = modeFor(req.entry.id)
  const regions = regionsFor(req.entry, ctx, req.pinned)
  const token = await ctx.token()

  const done = await mapLimit(chunk(input.texts, mode.batch), async batch => {
    const res = await withRegions(regions, loc => ctx.fetch(
      modelUrl(ctx.project, loc, req.entry.publisher, req.entry.id, mode.method),
      {
        method: "POST",
        headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
        body: JSON.stringify(mode.payload(batch, input)),
        signal: req.signal,
      },
    ), { signal: req.signal })
    if (!res.ok) return { error: { status: res.status, message: upstreamMessage(await res.text()) } }
    const embeddings = mode.read(await res.json(), batch.length)
    return embeddings ? { embeddings } : { error: { status: 502, message: "malformed embedding response" } }
  })
  if (done.error) return errorResponse("openai", done.error.status, done.error.message)

  const total = done.embeddings.reduce((sum, e) => sum + e.tokens, 0)
  return Response.json({
    object: "list",
    data: done.embeddings.map(({ values }, index) => ({
      object: "embedding",
      index,
      embedding: input.encodingFormat === "base64" ? toBase64(values) : values,
    })),
    model: req.apiId,
    usage: { prompt_tokens: total, total_tokens: total },
  })
}

function modeFor(id) {
  if (id.startsWith("gemini-embedding-2")) return EMBED_CONTENT
  if (id.startsWith("multimodalembedding")) return MULTIMODAL
  // gemini-embedding-001 accepts one instance per :predict call.
  if (id.startsWith("gemini-embedding")) return { ...TEXT_PREDICT, batch: 1 }
  return TEXT_PREDICT
}

function parseBody(body) {
  if (!body || typeof body !== "object") return { error: "request body must be a JSON object" }
  const { input, encoding_format: encodingFormat = "float", dimensions, task_type: taskType } = body
  if (Array.isArray(input) && input.some(x => typeof x !== "string")) {
    return { error: "token array input is not supported; send strings" }
  }
  const texts = typeof input === "string" ? [input] : input
  if (!Array.isArray(texts) || texts.length === 0) {
    return { error: "input must be a string or a non-empty array of strings" }
  }
  if (encodingFormat !== "float" && encodingFormat !== "base64") {
    return { error: "encoding_format must be float or base64" }
  }
  if (dimensions !== undefined && !(Number.isInteger(dimensions) && dimensions > 0)) {
    return { error: "dimensions must be a positive integer" }
  }
  if (taskType !== undefined && typeof taskType !== "string") return { error: "task_type must be a string" }
  return { texts, encodingFormat, dimensions, taskType }
}

function predictions(json, n, pick) {
  const preds = json?.predictions
  if (!Array.isArray(preds) || preds.length !== n) return null
  const out = preds.map(pick)
  return out.every(item => Array.isArray(item.values)) ? out : null
}

// Bounded concurrency; results keep input order and no new work starts after a failure.
async function mapLimit(items, fn) {
  const results = new Array(items.length)
  let next = 0
  let error
  const worker = async () => {
    while (!error && next < items.length) {
      const i = next++
      const r = await fn(items[i])
      if (r.error) error = r.error
      else results[i] = r.embeddings
    }
  }
  await Promise.all(Array.from({ length: Math.min(CONCURRENCY, items.length) }, worker))
  return error ? { error } : { embeddings: results.flat() }
}

function chunk(items, size) {
  const out = []
  for (let i = 0; i < items.length; i += size) out.push(items.slice(i, i + size))
  return out
}

// OpenAI base64 embeddings are the raw little-endian float32 bytes.
function toBase64(values) {
  const buf = Buffer.alloc(values.length * 4)
  values.forEach((v, i) => buf.writeFloatLE(v, i * 4))
  return buf.toString("base64")
}

function upstreamMessage(text) {
  try {
    const message = JSON.parse(text)?.error?.message
    if (message) return message
  } catch {
    // not JSON; fall through to the raw text
  }
  return text.slice(0, 500) || "upstream request failed"
}
