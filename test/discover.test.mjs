import { describe, expect, test } from "bun:test"
import { mkdir, mkdtemp, rm, stat, utimes, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { CATALOG, discoveredFile } from "../lib/catalog.mjs"
import { classifyResponse, discover, isStale, listedFamily } from "../lib/discover.mjs"

const OK = [200, "{}"]
const THROTTLED = [429, "RESOURCE_EXHAUSTED"]
const NOT_FOUND = [404, "NOT_FOUND"]
const FORBIDDEN = [403, "PERMISSION_DENIED"]
const UNIMPLEMENTED = [501, "Operation is not implemented, or supported, or enabled."]
const TERMS = [400, "FAILED_PRECONDITION Request rejected in accordance with Anthropic's terms of use and policies [Entity type: Billing Account]."]
const TERMS_403 = [403, "Request rejected because your project has not accepted the Gemini terms of use."]
const REGION_ONLY = [400, "Publisher model 'deepseek-ai/deepseek-v3.2-maas' in region 'us-central1' is only available via global endpoint."]
const VALIDATION = [400, "INVALID_ARGUMENT: max_tokens must be at least 2"]

// Verdicts keyed by model id. A value is one [status, text] for every call, or {loc: [status, text], op: [status, text], "*": [status, text]}
// where keys are regions (global, us-central1, ...) or operations (countTokens, generateContent, chat, predict, get).
function table(map) {
  return (id, loc, op) => {
    const verdict = map[id]
    if (!verdict) return NOT_FOUND
    if (Array.isArray(verdict)) return verdict
    return verdict[op] ?? verdict[loc] ?? verdict["*"] ?? NOT_FOUND
  }
}

// Vertex operation a request probes: the model path suffix, or the OpenAI-compatible surface it went through.
function operation(pathname) {
  const match = pathname.match(/\/models\/[^/:]+:(\w+)$/)
  if (match) return match[1] === "rawPredict" ? "countTokens" : match[1]
  if (pathname.endsWith("/chat/completions")) return "chat"
  return "get"
}

// Model id a request probes: the publisher-model path segment, or body.model for rawPredict and openapi chat calls.
// Null for URLs without a model path, e.g. the Cloud TTS voices endpoint.
function probedId(pathname, body) {
  if (pathname.endsWith("/models/count-tokens:rawPredict") || pathname.endsWith("/chat/completions")) return body.model
  return pathname.match(/\/models\/([^/:]+)(?::\w+)?$/)?.[1] ?? null
}

function fakeCtx({ listing = () => ({}), verdict = () => NOT_FOUND, token = async () => "tok" } = {}) {
  const calls = []
  const respond = ([status, text]) => new Response(text, { status })
  const fetch = async (url, init = {}) => {
    const u = new URL(url)
    const body = init.body ? JSON.parse(init.body) : undefined
    const loc = u.host === "aiplatform.googleapis.com" ? "global" : u.host.replace("-aiplatform.googleapis.com", "")
    const op = operation(u.pathname)
    const listed = u.pathname.match(/^\/v1beta1\/publishers\/([^/]+)\/models$/)
    const id = listed ? null : probedId(u.pathname, body)
    calls.push({ method: init.method ?? "GET", host: u.host, path: u.pathname, loc, op, id, body })
    if (u.host === "texttospeech.googleapis.com") return respond(verdict("chirp3-hd", "global", op))
    if (listed) return Response.json(await listing(listed[1], loc, u.searchParams.get("pageToken")))
    return respond(verdict(id, loc, op))
  }
  return { project: "proj-x", token, fetch, log: () => {}, calls }
}

async function withDir(run) {
  const root = await mkdtemp(join(tmpdir(), "vertex-discover-"))
  try {
    return await run(join(root, "config", "magpie"))
  } finally {
    await rm(root, { recursive: true, force: true })
  }
}

describe("classifyResponse", () => {
  test("only a 200 is available", () => {
    expect(classifyResponse(200)).toEqual({ available: true })
    expect(classifyResponse(503, "upstream error")).toEqual({ available: false, reason: "http-503" })
    expect(classifyResponse(501, "Operation is not implemented")).toEqual({ available: false, reason: "not-implemented" })
  })

  test("a quota 429 is throttled, any other 429 keeps the status", () => {
    expect(classifyResponse(429, "RESOURCE_EXHAUSTED: quota exceeded")).toEqual({ available: false, reason: "throttled" })
    expect(classifyResponse(429, "<html>slow down</html>")).toEqual({ available: false, reason: "http-429" })
  })

  test("404 and 403 are unavailable with a reason", () => {
    expect(classifyResponse(404, "NOT_FOUND")).toEqual({ available: false, reason: "not-found" })
    expect(classifyResponse(403, "PERMISSION_DENIED")).toEqual({ available: false, reason: "forbidden" })
  })

  test("the Anthropic terms gate is unavailable as terms", () => {
    expect(classifyResponse(...TERMS)).toEqual({ available: false, reason: "terms" })
    expect(classifyResponse(...TERMS_403)).toEqual({ available: false, reason: "terms" })
  })

  test("a region rejection is unavailable as region-only", () => {
    expect(classifyResponse(400, "Publisher Model is not available in region us-east5")).toEqual({ available: false, reason: "region-only" })
    expect(classifyResponse(400, "This model is only available via global endpoint")).toEqual({ available: false, reason: "region-only" })
  })

  test("a parameter-validation 400 means the model accepted the request", () => {
    expect(classifyResponse(...VALIDATION)).toEqual({ available: true })
  })

  test("other 400s and statuses are unavailable with the status", () => {
    expect(classifyResponse(400, "Not a chat model")).toEqual({ available: false, reason: "rejected" })
    // live-verified: the omni models answer INVALID_ARGUMENT for generateContent
    expect(classifyResponse(400, '{"status":"INVALID_ARGUMENT","message":"gemini-omni-flash-preview is only supported in the Interactions API"}'))
      .toEqual({ available: false, reason: "rejected" })
    // live-verified: the phantom regions answer FAILED_PRECONDITION for everything
    expect(classifyResponse(400, "Precondition check failed.")).toEqual({ available: false, reason: "rejected" })
    expect(classifyResponse(401, "UNAUTHENTICATED")).toEqual({ available: false, reason: "http-401" })
  })
})

describe("listedFamily", () => {
  test("classifies chat and media ids by name", () => {
    expect(listedFamily("google", "gemini-2.5-flash")).toBe("gemini")
    expect(listedFamily("google", "gemini-2.5-flash-image")).toBe("gemini-image")
    expect(listedFamily("google", "gemini-2.5-flash-tts")).toBe("gemini-tts")
    expect(listedFamily("google", "gemini-3.1-flash-tts-preview")).toBe("gemini-tts")
    expect(listedFamily("google", "gemini-embedding-2")).toBe("embedding")
    expect(listedFamily("google", "text-embedding-005")).toBe("embedding")
    expect(listedFamily("google", "veo-3.1-lite-generate-001")).toBe("veo")
    expect(listedFamily("google", "lyria-002")).toBe("lyria")
    expect(listedFamily("anthropic", "claude-haiku-4-5")).toBe("claude")
  })

  test("classifies partner models as maas", () => {
    expect(listedFamily("google", "gemma-4-26b-a4b-it-maas")).toBe("maas")
    expect(listedFamily("deepseek-ai", "deepseek-v3.2-maas")).toBe("maas")
    expect(listedFamily("xai", "grok-4.20-non-reasoning")).toBe("maas")
    expect(listedFamily("qwen", "qwen3-embedding")).toBe("maas")
  })

  test("excludes Live, WebSocket and non-candidate ids", () => {
    expect(listedFamily("google", "gemini-3.8-live")).toBeNull()
    expect(listedFamily("google", "gemini-live-2.5-flash-native-audio")).toBeNull()
    expect(listedFamily("google", "gemini-3.5-transcribe-live-preview")).toBeNull()
    expect(listedFamily("google", "imagen-4.0-generate-001")).toBeNull()
    expect(listedFamily("meta", "llama2")).toBeNull()
  })
})

describe("isStale", () => {
  test("a missing file is stale", async () => {
    await withDir(async (dir) => {
      expect(isStale(join(dir, "nope.json"))).toBe(true)
    })
  })

  test("uses the file mtime against the ttl", async () => {
    await withDir(async (dir) => {
      const file = join(dir, "catalog.json")
      await mkdir(dir, { recursive: true })
      await writeFile(file, "{}")
      expect(isStale(file)).toBe(false)
      const old = new Date(Date.now() - 48 * 3600_000)
      await utimes(file, old, old)
      expect(isStale(file)).toBe(true)
      expect(isStale(file, 72)).toBe(false)
    })
  })
})

describe("discover", () => {
  test("requires a dir", async () => {
    await expect(discover(fakeCtx(), {})).rejects.toThrow("dir is required")
  })

  test("classifies verdicts per region and writes the file", async () => {
    await withDir(async (dir) => {
      const ctx = fakeCtx({
        verdict: table({
          "gemini-2.5-flash": { global: OK, "us-central1": THROTTLED },
          "gemini-2.5-pro": { "*": FORBIDDEN },
          "claude-opus-4-6": { "*": TERMS },
          "text-embedding-005": { "us-central1": VALIDATION },
        }),
      })
      const data = await discover(ctx, { dir, retryDelay: 0 })

      const flash = data.models.find((m) => m.id === "gemini-2.5-flash")
      expect(flash.regions).toEqual(["global"])
      expect(data.models.find((m) => m.id === "text-embedding-005").regions).toEqual(["us-central1"])
      expect(data.unavailable).toContainEqual({ id: "gemini-2.5-pro", publisher: "google", family: "gemini", reason: "forbidden" })
      expect(data.unavailable).toContainEqual({ id: "claude-opus-4-6", publisher: "anthropic", family: "claude", reason: "terms" })

      const onDisk = JSON.parse(await Bun.file(discoveredFile(dir, "proj-x")).text())
      expect(onDisk).toEqual(data)
    })
  })

  test("a throttled region retries once and only a recovery counts", async () => {
    await withDir(async (dir) => {
      const attempts = new Map()
      const ctx = fakeCtx({
        verdict: (id, loc, op) => {
          const key = `${id}@${loc}:${op}`
          const n = (attempts.get(key) ?? 0) + 1
          attempts.set(key, n)
          return id === "gemini-2.5-flash" && loc === "global" && op === "countTokens" ? (n === 1 ? THROTTLED : OK) : OK
        },
      })
      const data = await discover(ctx, { dir, retryDelay: 0 })
      const counts = ctx.calls.filter((c) => c.id === "gemini-2.5-flash" && c.loc === "global" && c.op === "countTokens").length
      expect(counts).toBe(2)
      expect(data.models.find((m) => m.id === "gemini-2.5-flash").regions).toContain("global")
    })
  })

  test("a 5xx region stays unavailable after the retry, as not-implemented for 501", async () => {
    await withDir(async (dir) => {
      const ctx = fakeCtx({ verdict: () => UNIMPLEMENTED })
      const data = await discover(ctx, { dir, retryDelay: 0 })
      expect(data.models.find((m) => m.id === "gemini-2.5-flash")).toBeUndefined()
      expect(data.unavailable).toContainEqual({ id: "gemini-2.5-flash", publisher: "google", family: "gemini", reason: "not-implemented" })
      const counts = ctx.calls.filter((c) => c.id === "gemini-2.5-flash" && c.loc === "global" && c.op === "countTokens").length
      expect(counts).toBe(2)
    })
  })

  test("a network error retries once and a recovery counts", async () => {
    await withDir(async (dir) => {
      const inner = fakeCtx({ verdict: table({ "gemini-2.5-flash": { global: OK } }) })
      let failed = false
      const ctx = {
        ...inner,
        fetch: async (url, init) => {
          if (!failed && String(url).includes(":countTokens")) {
            failed = true
            throw new TypeError("certificate verification error")
          }
          return inner.fetch(url, init)
        },
      }
      const data = await discover(ctx, { dir, retryDelay: 0 })
      expect(data.models.find((m) => m.id === "gemini-2.5-flash").regions).toContain("global")
    })
  })

  test("a CATALOG model with only transient failures keeps its catalog regions and is flagged", async () => {
    await withDir(async (dir) => {
      const ctx = fakeCtx({ verdict: () => THROTTLED })
      const data = await discover(ctx, { dir, retryDelay: 0 })
      const regions = CATALOG.find((e) => e.id === "gemini-2.5-flash").regions
      expect(data.models).toContainEqual({ id: "gemini-2.5-flash", publisher: "google", family: "gemini", regions, transient: true })
      expect(data.unavailable.find((u) => u.id === "gemini-2.5-flash")).toBeUndefined()
    })
  })

  test("a transient gate failure flags a CATALOG model instead of dropping it", async () => {
    await withDir(async (dir) => {
      const ctx = fakeCtx({
        verdict: table({ "gemini-2.5-flash": { countTokens: { global: OK, "*": NOT_FOUND }, generateContent: THROTTLED } }),
      })
      const data = await discover(ctx, { dir, retryDelay: 0 })
      const regions = CATALOG.find((e) => e.id === "gemini-2.5-flash").regions
      expect(data.models).toContainEqual({ id: "gemini-2.5-flash", publisher: "google", family: "gemini", regions, transient: true })
    })
  })

  test("a discovered-only model with only transient failures stays unavailable", async () => {
    await withDir(async (dir) => {
      const ctx = fakeCtx({
        listing: (publisher) => (publisher === "google" ? { publisherModels: [{ name: "publishers/google/models/gemini-9-flash" }] } : {}),
        verdict: () => THROTTLED,
      })
      const data = await discover(ctx, { dir, retryDelay: 0 })
      expect(data.unavailable).toContainEqual({ id: "gemini-9-flash", publisher: "google", family: "gemini", reason: "throttled" })
      expect(data.models.find((m) => m.id === "gemini-9-flash")).toBeUndefined()
    })
  })

  test("a permanent reason alongside transient ones keeps the model unavailable", async () => {
    await withDir(async (dir) => {
      const ctx = fakeCtx({ verdict: (id, loc, op) => (loc === "global" ? NOT_FOUND : THROTTLED) })
      const data = await discover(ctx, { dir, retryDelay: 0 })
      expect(data.models.find((m) => m.id === "gemini-2.5-flash")).toBeUndefined()
      expect(data.unavailable).toContainEqual({ id: "gemini-2.5-flash", publisher: "google", family: "gemini", reason: "not-found, throttled" })
    })
  })

  test("a region-only rejection does not block the transient rescue", async () => {
    await withDir(async (dir) => {
      const ctx = fakeCtx({
        listing: (publisher, loc) =>
          publisher === "deepseek-ai" && loc === "us-central1"
            ? { publisherModels: [{ name: "publishers/deepseek-ai/models/deepseek-v3.2-maas", versionId: "default" }] }
            : {},
        verdict: table({ "deepseek-ai/deepseek-v3.2-maas": { chat: { global: THROTTLED, "us-central1": REGION_ONLY } } }),
      })
      const data = await discover(ctx, { dir, retryDelay: 0 })
      expect(data.models).toContainEqual({ id: "deepseek-v3.2-maas", publisher: "deepseek-ai", family: "maas", regions: ["global"], transient: true })
      expect(data.unavailable.find((u) => u.id === "deepseek-v3.2-maas")).toBeUndefined()
    })
  })

  test("region-only everywhere with no transient reason stays unavailable", async () => {
    await withDir(async (dir) => {
      const ctx = fakeCtx({ verdict: () => REGION_ONLY })
      const data = await discover(ctx, { dir, retryDelay: 0 })
      expect(data.models.find((m) => m.id === "gemini-2.5-flash")).toBeUndefined()
      expect(data.unavailable).toContainEqual({ id: "gemini-2.5-flash", publisher: "google", family: "gemini", reason: "region-only" })
    })
  })

  test("a model-level gate turns a countTokens pass into unavailable", async () => {
    await withDir(async (dir) => {
      const ctx = fakeCtx({
        listing: (publisher) => (publisher === "google" ? { publisherModels: [{ name: "publishers/google/models/gemini-4-argon" }] } : {}),
        verdict: table({ "gemini-4-argon": { countTokens: OK, generateContent: TERMS_403 } }),
      })
      const data = await discover(ctx, { dir, retryDelay: 0 })
      expect(data.models.find((m) => m.id === "gemini-4-argon")).toBeUndefined()
      expect(data.unavailable).toContainEqual({ id: "gemini-4-argon", publisher: "google", family: "gemini", reason: "terms" })
      const generates = ctx.calls.filter((c) => c.op === "generateContent" && c.id === "gemini-4-argon")
      expect(generates.length).toBe(1)
      expect(generates[0].body.generationConfig).toEqual({ maxOutputTokens: 1 })
    })
  })

  test("a persistent quota 429 on generate marks the model throttled", async () => {
    await withDir(async (dir) => {
      const ctx = fakeCtx({
        listing: (publisher) => (publisher === "google" ? { publisherModels: [{ name: "publishers/google/models/gemini-3.7-flash" }] } : {}),
        verdict: table({ "gemini-3.7-flash": { countTokens: OK, generateContent: THROTTLED } }),
      })
      const data = await discover(ctx, { dir, retryDelay: 0 })
      expect(data.unavailable).toContainEqual({ id: "gemini-3.7-flash", publisher: "google", family: "gemini", reason: "throttled" })
      expect(ctx.calls.filter((c) => c.op === "generateContent" && c.id === "gemini-3.7-flash").length).toBe(2)
    })
  })

  test("a 200 on the retried generate keeps the model", async () => {
    await withDir(async (dir) => {
      let generateAttempts = 0
      const ctx = fakeCtx({
        listing: (publisher) => (publisher === "google" ? { publisherModels: [{ name: "publishers/google/models/gemini-3.7-flash" }] } : {}),
        verdict: (id, loc, op) => {
          if (id === "gemini-3.7-flash" && op === "generateContent") return ++generateAttempts === 1 ? THROTTLED : OK
          return OK
        },
      })
      const data = await discover(ctx, { dir, retryDelay: 0 })
      expect(data.models.find((m) => m.id === "gemini-3.7-flash").regions).toContain("global")
    })
  })

  test("gemini-tts proves itself with one audio request, image models skip the check", async () => {
    await withDir(async (dir) => {
      const ctx = fakeCtx({
        listing: (publisher) =>
          publisher === "google"
            ? {
                publisherModels: [
                  { name: "publishers/google/models/gemini-9-voice-tts" },
                  { name: "publishers/google/models/gemini-9-draw-image" },
                ],
              }
            : {},
        verdict: table({ "gemini-9-voice-tts": OK, "gemini-9-draw-image": OK }),
      })
      const data = await discover(ctx, { dir, retryDelay: 0 })
      expect(data.models.find((m) => m.id === "gemini-9-voice-tts")).toBeTruthy()
      expect(data.models.find((m) => m.id === "gemini-9-draw-image")).toBeTruthy()
      const generates = ctx.calls.filter((c) => c.op === "generateContent")
      expect(generates.length).toBe(1)
      expect(generates[0].id).toBe("gemini-9-voice-tts")
      expect(generates[0].body.generationConfig.responseModalities).toEqual(["AUDIO"])
      expect(generates[0].body.generationConfig.speechConfig.voiceConfig.prebuiltVoiceConfig.voiceName).toBe("Kore")
    })
  })

  test("newly discovered ids are limited to served families", async () => {
    await withDir(async (dir) => {
      const ctx = fakeCtx({
        listing: (publisher) =>
          publisher === "google"
            ? {
                publisherModels: [
                  { name: "publishers/google/models/gemini-9-flash" },
                  { name: "publishers/google/models/veo-9-generate-001" },
                  { name: "publishers/google/models/lyria-3-pro-preview" },
                  { name: "publishers/google/models/gemini-9-live" },
                ],
              }
            : {},
        verdict: table({ "gemini-9-flash": OK, "veo-9-generate-001": OK, "lyria-3-pro-preview": OK, "gemini-9-live": OK }),
      })
      const data = await discover(ctx, { dir, retryDelay: 0 })
      const ids = data.models.map((m) => m.id)
      expect(ids).toContain("gemini-9-flash")
      expect(ids).toContain("veo-9-generate-001")
      expect(ids).not.toContain("lyria-3-pro-preview")
      expect(ids).not.toContain("gemini-9-live")
    })
  })

  test("gemini embeddings are probed with embedContent, other text embeddings with predict", async () => {
    await withDir(async (dir) => {
      const ctx = fakeCtx({
        verdict: table({ "gemini-embedding-2": { global: OK, "us-central1": OK }, "text-embedding-005": { "us-central1": OK } }),
      })
      const data = await discover(ctx, { dir, retryDelay: 0 })
      const geminiCalls = ctx.calls.filter((c) => c.id === "gemini-embedding-2" && c.loc === "global")
      expect(geminiCalls.every((c) => c.op === "embedContent")).toBe(true)
      expect(geminiCalls[0].body).toEqual({ content: { parts: [{ text: "hi" }] } })
      const textCalls = ctx.calls.filter((c) => c.id === "text-embedding-005" && c.loc === "us-central1")
      expect(textCalls.every((c) => c.op === "predict")).toBe(true)
      expect(textCalls[0].body).toEqual({ instances: [{ content: "hi" }] })
      expect(data.models.find((m) => m.id === "gemini-embedding-2").regions).toContain("global")
    })
  })

  test("the written file has the documented shape", async () => {
    await withDir(async (dir) => {
      const data = await discover(fakeCtx({ verdict: table({ "gemini-2.5-flash": { global: OK } }) }), { dir, retryDelay: 0 })
      expect(Object.keys(data).sort()).toEqual(["discoveredAt", "models", "project", "unavailable"])
      expect(data.project).toBe("proj-x")
      expect(new Date(data.discoveredAt).toISOString()).toBe(data.discoveredAt)
      const model = data.models.find((m) => m.id === "gemini-2.5-flash")
      expect(Object.keys(model)).toEqual(["id", "publisher", "family", "regions"])
      const gone = data.unavailable[0]
      expect(Object.keys(gone)).toEqual(["id", "publisher", "family", "reason"])
      expect((await stat(discoveredFile(dir, "proj-x"))).isFile()).toBe(true)
    })
  })

  test("a MaaS model is probed only where it is listed, plus global", async () => {
    await withDir(async (dir) => {
      const ctx = fakeCtx({
        listing: (publisher, loc) =>
          publisher === "deepseek-ai" && loc === "us-central1"
            ? { publisherModels: [{ name: "publishers/deepseek-ai/models/deepseek-v3.2-maas", versionId: "default" }] }
            : {},
        verdict: table({ "deepseek-ai/deepseek-v3.2-maas": OK }),
      })
      const data = await discover(ctx, { dir, retryDelay: 0 })
      const probed = ctx.calls.filter((c) => c.body?.model === "deepseek-ai/deepseek-v3.2-maas").map((c) => c.loc).sort()
      expect(probed).toEqual(["global", "us-central1"])
      expect(data.models.find((m) => m.id === "deepseek-v3.2-maas").regions).toEqual(["global", "us-central1"])
    })
  })

  test("chirp TTS is one global check against the Cloud TTS voices endpoint", async () => {
    await withDir(async (dir) => {
      const ctx = fakeCtx({ verdict: table({ "chirp3-hd": OK }) })
      const data = await discover(ctx, { dir, retryDelay: 0 })
      const voices = ctx.calls.filter((c) => c.host === "texttospeech.googleapis.com")
      expect(voices.length).toBe(1)
      expect(voices[0].method).toBe("GET")
      expect(data.models.find((m) => m.id === "chirp3-hd").regions).toEqual(["global"])
    })
  })

  test("follows nextPageToken and probes ids found on later pages", async () => {
    await withDir(async (dir) => {
      const ctx = fakeCtx({
        listing: (publisher, loc, pageToken) => {
          if (publisher !== "google" || loc !== "global") return {}
          return pageToken
            ? { publisherModels: [{ name: "publishers/google/models/gemini-future-9-flash" }] }
            : { publisherModels: [{ name: "publishers/google/models/gemini-2.5-flash" }], nextPageToken: "page2" }
        },
        verdict: table({ "gemini-future-9-flash": { global: OK } }),
      })
      const data = await discover(ctx, { dir, retryDelay: 0 })
      expect(data.models).toContainEqual({ id: "gemini-future-9-flash", publisher: "google", family: "gemini", regions: ["global"] })
    })
  })

  test("a listed Claude version becomes an @version id and is probed with it", async () => {
    await withDir(async (dir) => {
      const ctx = fakeCtx({
        listing: (publisher, loc) =>
          publisher === "anthropic" && loc === "us-central1"
            ? { publisherModels: [{ name: "publishers/anthropic/models/claude-haiku-5-5", versionId: "20261005" }] }
            : {},
        verdict: table({ "claude-haiku-5-5@20261005": { "us-central1": OK } }),
      })
      const data = await discover(ctx, { dir, retryDelay: 0 })
      expect(data.models).toContainEqual({ id: "claude-haiku-5-5@20261005", publisher: "anthropic", family: "claude", regions: ["us-central1"] })
      expect(ctx.calls.some((c) => c.body?.model === "claude-haiku-5-5@20261005" && c.loc === "us-central1")).toBe(true)
    })
  })

  test("an expired login stops the run instead of marking every model unavailable", async () => {
    await withDir(async (dir) => {
      const ctx = fakeCtx({
        token: async () => {
          throw Object.assign(new Error("sign in again"), { signIn: "expired" })
        },
      })
      await expect(discover(ctx, { dir })).rejects.toThrow("sign in again")
    })
  })

  test("never runs more probes at once than the concurrency option", async () => {
    await withDir(async (dir) => {
      const inner = fakeCtx({ verdict: table({ "gemini-2.5-flash": { global: OK } }) })
      let inFlight = 0
      let peak = 0
      const ctx = {
        ...inner,
        fetch: async (url, init) => {
          inFlight++
          peak = Math.max(peak, inFlight)
          await Promise.resolve()
          try {
            return await inner.fetch(url, init)
          } finally {
            inFlight--
          }
        },
      }
      await discover(ctx, { dir, concurrency: 3, retryDelay: 0 })
      expect(peak).toBeGreaterThanOrEqual(2)
      expect(peak).toBeLessThanOrEqual(3)
    })
  })

  test("reports list, probe and gate progress", async () => {
    await withDir(async (dir) => {
      const events = []
      await discover(fakeCtx(), { dir, retryDelay: 0, onProgress: (e) => events.push(e) })
      for (const phase of ["list", "probe", "gate"]) {
        const events_ = events.filter((e) => e.phase === phase)
        expect(events_.length).toBeGreaterThan(0)
        expect(events_.at(-1).done).toBe(events_.at(-1).total)
      }
      expect(events.find((e) => e.phase === "probe")).toMatchObject({ id: expect.any(String), loc: expect.any(String) })
    })
  })
})
