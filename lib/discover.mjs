// Probes which Vertex models this project can call, so the plugin only offers models the account reaches.
import { statSync } from "node:fs"
import { mkdir, rename, writeFile } from "node:fs/promises"
import { dirname } from "node:path"
import { CATALOG, discoveredFile, orderRegions } from "./catalog.mjs"
import { REGIONS, hostFor, modelUrl, openapiUrl } from "./regions.mjs"

const PUBLISHERS = ["google", "anthropic", "deepseek-ai", "qwen", "zai-org", "moonshotai", "minimaxai", "openai", "xai", "meta", "mistralai", "ai21"]
const CHIRP_VOICES_URL = "https://texttospeech.googleapis.com/v1/voices?languageCode=en-US"
const MAX_LIST_PAGES = 20
const TIMEOUT_MS = 60_000
const RETRY_DELAY_MS = 2_000

const CANDIDATE_NAME = /^(gemini-|grok-|claude-|veo-|lyria-)|-maas$|embedding|-tts\b/
const LIVE_NAME = /live|native-audio/
const TERMS = /terms of use|terms and policies|billing account|terms of service/i
const REGION = /only available|not available in|not servable|not supported in this region|only via global/i
const QUOTA = /RESOURCE_EXHAUSTED|quota/i
// Only a 400 that rejects a parameter (live-verified: "max_tokens must be at least 2") means the endpoint accepted
// the model; status INVALID_ARGUMENT alone also covers wrong-API rejections ("only supported in the Interactions API").
const VALIDATION = /must be|is required|unknown name|unrecognized/i

// Families the route handlers serve with the standard shape; other discovered ids stay out of the catalog file.
const DISCOVERABLE = new Set(["gemini", "gemini-image", "gemini-tts", "embedding", "maas", "claude", "veo"])
// Families whose cheap region probe (countTokens) passes gated models, so one real generation decides per model.
const GATED = new Set(["gemini", "gemini-tts"])

// Regions a rescued model falls back to: a known model must not vanish for the TTL over a probe-time blip.
const CATALOG_REGIONS = new Map(CATALOG.map((e) => [`${e.publisher}/${e.id}`, e.regions]))

// Reasons a later refresh can clear: quota throttling, network errors, server errors (501 UNIMPLEMENTED is permanent).
function isTransient(reason) {
  if (reason === "throttled" || reason === "network") return true
  if (!reason.startsWith("http-")) return false
  const status = Number(reason.slice(5))
  return status === 429 || (status >= 500 && status !== 501)
}

// Family of a publisher-listed model id, or null when it is not a chat or media model worth probing.
export function listedFamily(publisher, id) {
  if (LIVE_NAME.test(id) || !CANDIDATE_NAME.test(id)) return null
  if (publisher === "anthropic" || id.startsWith("claude-")) return "claude"
  if (publisher !== "google") return "maas"
  if (id.startsWith("veo-")) return "veo"
  if (id.startsWith("lyria-")) return "lyria"
  if (id.includes("embedding")) return "embedding"
  if (/-tts\b/.test(id)) return "gemini-tts"
  if (id.endsWith("-maas")) return "maas"
  if (/image|nano-banana/.test(id)) return "gemini-image"
  return "gemini"
}

// Verdict for one probe response. Only a 200 makes a region available; every regional endpoint answers 501/502
// for models it does not serve, so 429-quota and 5xx are decided by the retry, not counted on sight.
export function classifyResponse(status, text = "") {
  if (status === 200) return { available: true }
  if (status === 429) return { available: false, reason: QUOTA.test(text) ? "throttled" : "http-429" }
  if (TERMS.test(text)) return { available: false, reason: "terms" }
  if (status === 403) return { available: false, reason: "forbidden" }
  if (status === 404) return { available: false, reason: "not-found" }
  if (status === 400) {
    if (REGION.test(text)) return { available: false, reason: "region-only" }
    if (VALIDATION.test(text)) return { available: true } // the endpoint parsed a valid request and rejected a parameter
    return { available: false, reason: "rejected" }
  }
  if (status === 501) return { available: false, reason: "not-implemented" }
  return { available: false, reason: `http-${status}` }
}

export function isStale(file, ttlHours = 24) {
  try {
    return Date.now() - statSync(file).mtimeMs > ttlHours * 3600_000
  } catch {
    return true
  }
}

export async function discover(ctx, { dir, concurrency = 16, onProgress, retryDelay = RETRY_DELAY_MS } = {}) {
  if (!dir) throw new Error("discover: dir is required")

  const listings = PUBLISHERS.flatMap((publisher) => REGIONS.map((loc) => ({ publisher, loc })))
  const listed = new Map()
  const listedIn = new Map()
  let listedDone = 0
  await mapLimit(listings, concurrency, async ({ publisher, loc }) => {
    for (const m of await listPublisher(ctx, publisher, loc)) {
      const name = m.name?.split("/").pop()
      const family = name && listedFamily(publisher, name)
      if (!family) continue
      const id = family === "claude" && m.versionId && m.versionId !== "default" ? `${name}@${m.versionId}` : name
      const key = `${publisher}/${id}`
      if (!listed.has(key)) listed.set(key, { id, publisher, family })
      if (!listedIn.has(key)) listedIn.set(key, new Set())
      listedIn.get(key).add(loc)
    }
    onProgress?.({ phase: "list", done: ++listedDone, total: listings.length })
  })

  const candidates = new Map()
  for (const e of CATALOG) candidates.set(`${e.publisher}/${e.id}`, { id: e.id, publisher: e.publisher, family: e.family })
  for (const [key, c] of listed) if (!candidates.has(key) && discoverable(c)) candidates.set(key, c)

  const probes = []
  for (const [key, c] of candidates) {
    for (const loc of probeRegions(c, listedIn.get(key))) probes.push({ key, c, loc })
  }
  const verdicts = new Map()
  let probed = 0
  await mapLimit(probes, concurrency, async ({ key, c, loc }) => {
    const verdict = await probe(ctx, probeRequest(ctx.project, c, loc), retryDelay)
    if (!verdicts.has(key)) verdicts.set(key, new Map())
    verdicts.get(key).set(loc, verdict)
    onProgress?.({ phase: "probe", done: ++probed, total: probes.length, id: c.id, loc })
  })

  // One real generation per gated family at its first region: countTokens answers 200 even for models whose
  // generateContent is refused (terms, quota, wrong API surface).
  const gated = [...candidates].filter(([key, c]) => GATED.has(c.family))
  const gates = new Map()
  let gatedDone = 0
  await mapLimit(gated, concurrency, async ([key, c]) => {
    const first = firstRegion(verdicts.get(key))
    gates.set(key, first ? await probe(ctx, gateRequest(ctx.project, c, first), retryDelay) : { available: false, reason: "" })
    onProgress?.({ phase: "gate", done: ++gatedDone, total: gated.length })
  })

  const models = []
  const unavailable = []
  for (const [key, c] of candidates) {
    const results = verdicts.get(key) ?? new Map()
    const gate = gates.get(key)
    const gateReason = gate && !gate.available && gate.reason ? gate.reason : null
    const open = gateReason ? new Set() : new Set([...results].filter(([, v]) => v.available).map(([loc]) => loc))
    const { id, publisher, family } = c
    if (open.size) {
      // Regions with a 200 stand on their own; transient failures elsewhere are simply not listed.
      models.push({ id, publisher, family, regions: orderRegions(REGIONS.filter((loc) => open.has(loc))) })
      continue
    }
    const reasons = gateReason ? [gateReason] : [...results.values()].map((v) => v.reason)
    const catalogRegions = CATALOG_REGIONS.get(key)
    // region-only only says "not here, try another region", so it is neutral: decide on the remaining reasons.
    const decisive = reasons.filter((reason) => reason !== "region-only")
    if (decisive.length && decisive.every(isTransient) && catalogRegions) {
      models.push({ id, publisher, family, regions: catalogRegions, transient: true })
    } else {
      unavailable.push({ id, publisher, family, reason: gateReason ?? distinctReasons(results) })
    }
  }

  const data = { project: ctx.project, discoveredAt: new Date().toISOString(), models, unavailable }
  const file = discoveredFile(dir, ctx.project)
  await mkdir(dirname(file), { recursive: true })
  const tmp = `${file}.${process.pid}.tmp`
  await writeFile(tmp, `${JSON.stringify(data, null, 2)}\n`)
  await rename(tmp, file)
  return data
}

// A discovered id joins the catalog file only for served families; lyria only as lyria-002, Live ids never.
function discoverable(c) {
  if (!DISCOVERABLE.has(c.family)) return false
  return c.family !== "lyria" || c.id === "lyria-002"
}

// Chirp is a Cloud TTS voice, so it is one global check. MaaS models are only listed in some regions, so only those plus global are probed.
function probeRegions(c, listedRegions = new Set()) {
  if (c.family === "chirp-tts") return ["global"]
  if (c.family === "maas") return [...new Set(["global", ...listedRegions])]
  return REGIONS
}

// The cheapest request that proves a model is callable in one region: countTokens, a metadata GET, or a 1-token chat.
function probeRequest(project, c, loc) {
  const { id, publisher, family } = c
  const hi = [{ role: "user", content: "hi" }]
  if (family === "chirp-tts") return { method: "GET", url: CHIRP_VOICES_URL }
  if (family === "veo" || family === "lyria") {
    return { method: "GET", url: `https://${hostFor(loc)}/v1beta1/publishers/${publisher}/models/${id}` }
  }
  if (family === "maas") {
    return {
      method: "POST",
      url: openapiUrl(project, loc, "/chat/completions"),
      body: { model: `${publisher}/${id}`, messages: hi, max_tokens: 1 },
    }
  }
  if (family === "claude") {
    return {
      method: "POST",
      url: modelUrl(project, loc, "anthropic", "count-tokens", "rawPredict"),
      body: { anthropic_version: "vertex-2023-10-16", model: id, messages: hi },
    }
  }
  if (family === "embedding") {
    if (id.startsWith("gemini-embedding")) {
      return { method: "POST", url: modelUrl(project, loc, publisher, id, "embedContent"), body: { content: { parts: [{ text: "hi" }] } } }
    }
    const instance = id.startsWith("multimodalembedding") ? { text: "hi" } : { content: "hi" }
    return { method: "POST", url: modelUrl(project, loc, publisher, id, "predict"), body: { instances: [instance] } }
  }
  return {
    method: "POST",
    url: modelUrl(project, loc, publisher, id, "countTokens"),
    body: { contents: [{ role: "user", parts: [{ text: "hi" }] }] },
  }
}

// The smallest real generation for a gated family: one output token for chat, one spoken word for TTS.
// Image models are skipped: their one-word request bills an image, and countTokens already proves reachability.
function gateRequest(project, c, loc) {
  const generationConfig = c.family === "gemini-tts"
    ? { responseModalities: ["AUDIO"], speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: "Kore" } } } }
    : { maxOutputTokens: 1 }
  return {
    method: "POST",
    url: modelUrl(project, loc, c.publisher, c.id, "generateContent"),
    body: { contents: [{ role: "user", parts: [{ text: "hi" }] }], generationConfig },
  }
}

function firstRegion(results) {
  if (!results) return null
  const open = [...results].filter(([, v]) => v.available).map(([loc]) => loc)
  return open.length ? orderRegions(REGIONS.filter((loc) => open.includes(loc)))[0] : null
}

function distinctReasons(results) {
  return [...new Set([...results.values()].map((v) => v.reason))].filter(Boolean).join(", ")
}

async function probe(ctx, req, retryDelay) {
  // An expired login is fatal for the whole run, not a per-model verdict.
  const token = await ctx.token()
  const attempt = async () => {
    try {
      const res = await ctx.fetch(req.url, {
        method: req.method,
        headers: { authorization: `Bearer ${token}`, ...(req.body && { "content-type": "application/json" }) },
        body: req.body && JSON.stringify(req.body),
        signal: AbortSignal.timeout(TIMEOUT_MS),
      })
      return { status: res.status, text: await res.text() }
    } catch {
      return null
    }
  }
  let out = await attempt()
  if (!out || out.status === 429 || out.status >= 500) {
    // Jitter so a burst of throttled probes does not land its retries in one cluster.
    if (retryDelay > 0) await new Promise((resolve) => setTimeout(resolve, retryDelay + Math.random() * retryDelay))
    out = await attempt()
  }
  if (!out) return { available: false, reason: "network" }
  return classifyResponse(out.status, out.text)
}

async function listPublisher(ctx, publisher, loc) {
  const models = []
  let pageToken
  for (let page = 0; page < MAX_LIST_PAGES; page++) {
    const query = new URLSearchParams({ pageSize: "200", listAllVersions: "true" })
    if (pageToken) query.set("pageToken", pageToken)
    const url = `https://${hostFor(loc)}/v1beta1/publishers/${publisher}/models?${query}`
    let body
    const token = await ctx.token()
    try {
      const res = await ctx.fetch(url, { headers: { authorization: `Bearer ${token}` }, signal: AbortSignal.timeout(TIMEOUT_MS) })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      body = await res.json()
    } catch (err) {
      ctx.log?.(`list ${publisher} ${loc}: ${err.message}`)
      break
    }
    models.push(...(body.publisherModels ?? []))
    pageToken = body.nextPageToken
    if (!pageToken) break
  }
  return models
}

// Runs fn over items with at most `limit` in flight; keeps results in item order and stops launching work after the first throw.
async function mapLimit(items, limit, fn) {
  const out = new Array(items.length)
  let next = 0
  let failed = null
  const worker = async () => {
    while (!failed && next < items.length) {
      const i = next++
      try {
        out[i] = await fn(items[i], i)
      } catch (err) {
        failed ??= err
        return
      }
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker))
  if (failed) throw failed
  return out
}
