import { describe, expect, test } from "bun:test"
import { mkdtemp, rm, writeFile, mkdir } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import {
  CATALOG,
  apiIdOf,
  configModels,
  discoveredFile,
  loadDiscovered,
  lookup,
  mergeDiscovered,
  modelsFor,
  npmFor,
  orderRegions,
  toModel,
} from "../lib/catalog.mjs"
import { BASE } from "../lib/regions.mjs"

// Every callable model in vertex-inventory.md section (b), excluding Live/WebSocket ones.
const CALLABLE = [
  "gemini-embedding-001",
  "gemini-embedding-2",
  "multimodalembedding",
  "text-embedding-005",
  "text-embedding-large-exp-03-07",
  "text-multilingual-embedding-002",
  "gemini-2.5-computer-use-preview-10-2025",
  "gemini-2.5-flash",
  "gemini-2.5-flash-lite",
  "gemini-2.5-pro",
  "gemini-3-flash-preview",
  "gemini-3.1-flash-lite",
  "gemini-3.1-pro-preview",
  "gemini-3.5-flash",
  "gemini-3.5-flash-lite",
  "gemini-3.5-transcribe-preview",
  "gemini-3.6-flash",
  "gemini-3.8-flash",
  "gemini-2.5-flash-image",
  "gemini-2.5-flash-tts",
  "gemini-2.5-pro-tts",
  "gemini-3.1-flash-tts-preview",
  "deepseek-ai/deepseek-ocr-maas",
  "deepseek-ai/deepseek-r1-0528-maas",
  "deepseek-ai/deepseek-v3.2-maas",
  "google/gemma-4-26b-a4b-it-maas",
  "minimaxai/minimax-m2-maas",
  "moonshotai/kimi-k2-thinking-maas",
  "openai/gpt-oss-120b-maas",
  "qwen/qwen3-235b-a22b-instruct-2507-maas",
  "qwen/qwen3-coder-480b-a35b-instruct-maas",
  "qwen/qwen3-next-80b-a3b-instruct-maas",
  "qwen/qwen3-next-80b-a3b-thinking-maas",
  "xai/grok-4.1-fast-non-reasoning",
  "xai/grok-4.1-fast-reasoning",
  "xai/grok-4.20-non-reasoning",
  "zai-org/glm-4.7-maas",
  "zai-org/glm-5-maas",
  "zai-org/glm-5.2-maas",
]

describe("CATALOG", () => {
  test("contains every callable inventory model", () => {
    for (const apiId of CALLABLE) {
      const found = lookup(apiId)
      expect(found).not.toBeNull()
      expect(found.entry.regions.length).toBeGreaterThan(0)
    }
  })

  test("contains the Veo, Lyria, Chirp and Claude entries", () => {
    for (const id of ["veo-2.0-generate-001", "veo-3.1-lite-generate-001", "lyria-002", "chirp3-hd", "claude-haiku-4-5@20251001", "claude-3-opus@20240229"]) {
      expect(lookup(id)).not.toBeNull()
    }
  })

  test("leaves out Live, Imagen, gated and metadata-only models", () => {
    for (const id of [
      "gemini-3.8-live",
      "gemini-live-2.5-flash-native-audio",
      "gemini-3.5-transcribe-live-preview",
      "imagen-4.0-generate-001",
      "gemini-3.7-flash",
      "gemini-4-argon",
      "lyria-3-pro-preview",
      "alphafold3-request",
    ]) {
      expect(lookup(id)).toBeNull()
    }
  })

  test("keys are unique", () => {
    expect(new Set(CATALOG.map((e) => e.key)).size).toBe(CATALOG.length)
  })

  test("every entry has regions in preferred order and sane limits", () => {
    for (const e of CATALOG) {
      expect(e.regions.length).toBeGreaterThan(0)
      expect(new Set(e.regions).size).toBe(e.regions.length)
      expect(e.regions).toEqual(orderRegions(e.regions))
      expect(e.context).toBeGreaterThan(0)
      expect(e.output).toBeGreaterThanOrEqual(0)
      expect(e.key).not.toContain("/")
      expect(e.key).not.toContain("@")
    }
  })

  test("embedding entries have no output and no outputs", () => {
    for (const e of CATALOG.filter((c) => c.family === "embedding")) {
      expect(e.output).toBe(0)
      expect(e.outputs).toEqual([])
    }
  })
})

describe("lookup", () => {
  test("finds a model by magpie key", () => {
    const { entry, pinned } = lookup("gemini-2.5-flash")
    expect(entry.id).toBe("gemini-2.5-flash")
    expect(pinned).toBeNull()
  })

  test("finds a MaaS model by its Vertex api id", () => {
    const { entry, pinned } = lookup("deepseek-ai/deepseek-v3.2-maas")
    expect(entry.key).toBe("deepseek-v3.2-maas")
    expect(pinned).toBeNull()
  })

  test("finds a versioned Claude model by its api id, which contains @", () => {
    const { entry, pinned } = lookup("claude-haiku-4-5@20251001")
    expect(entry.key).toBe("claude-haiku-4-5")
    expect(pinned).toBeNull()
  })

  test("pins a region that the entry lists", () => {
    expect(lookup("gemini-2.5-flash@us-east5")).toEqual({ entry: lookup("gemini-2.5-flash").entry, pinned: "us-east5" })
    expect(lookup("deepseek-ai/deepseek-v3.2-maas@global").pinned).toBe("global")
    expect(lookup("claude-haiku-4-5@global").pinned).toBe("global")
  })

  test("returns null for a region the entry does not list", () => {
    expect(lookup("gemini-2.5-flash@asia-east1")).toBeNull()
    expect(lookup("gemini-3.1-pro-preview@us-central1")).toBeNull()
  })

  test("returns null for unknown ids and non-strings", () => {
    expect(lookup("no-such-model")).toBeNull()
    expect(lookup("no-such-model@global")).toBeNull()
    expect(lookup(undefined)).toBeNull()
  })

  test("searches the list it is given", () => {
    const list = [{ ...lookup("gemini-2.5-pro").entry, regions: ["europe-west4"] }]
    expect(lookup("gemini-2.5-pro@europe-west4", list).pinned).toBe("europe-west4")
    expect(lookup("gemini-2.5-pro@global", list)).toBeNull()
  })
})

describe("apiIdOf and npmFor", () => {
  test("MaaS ids carry the publisher; other ids do not", () => {
    expect(apiIdOf(lookup("deepseek-v3.2-maas").entry)).toBe("deepseek-ai/deepseek-v3.2-maas")
    expect(apiIdOf(lookup("claude-haiku-4-5").entry)).toBe("claude-haiku-4-5@20251001")
    expect(apiIdOf(lookup("gemini-2.5-flash").entry)).toBe("gemini-2.5-flash")
  })

  test("maps each family to its npm package", () => {
    expect(npmFor("gemini")).toBe("@ai-sdk/google")
    expect(npmFor("gemini-image")).toBe("@ai-sdk/google")
    expect(npmFor("gemini-tts")).toBe("@ai-sdk/google")
    expect(npmFor("chirp-tts")).toBe("@ai-sdk/google")
    expect(npmFor("veo")).toBe("@ai-sdk/google")
    expect(npmFor("lyria")).toBe("@ai-sdk/google")
    expect(npmFor("claude")).toBe("@ai-sdk/anthropic")
    expect(npmFor("maas")).toBe("@ai-sdk/openai-compatible")
    expect(npmFor("embedding")).toBe("@ai-sdk/openai-compatible")
  })
})

describe("modelsFor", () => {
  test("lists every entry by key, without region variants by default", () => {
    const models = modelsFor({})
    expect(Object.keys(models)).toContain("gemini-2.5-flash")
    expect(Object.keys(models)).not.toContain("gemini-2.5-flash@us-east5")
    expect(Object.keys(models).length).toBe(CATALOG.length)
  })

  test("include keeps only the listed keys", () => {
    const models = modelsFor({ options: { include: ["gemini-2.5-flash", "text-embedding-005"] } })
    expect(Object.keys(models).sort()).toEqual(["gemini-2.5-flash", "text-embedding-005"])
  })

  test("include accepts /regex/ strings", () => {
    const models = modelsFor({ options: { include: ["/^gemini-2\\.5-/"] } })
    expect(Object.keys(models).length).toBeGreaterThan(1)
    for (const key of Object.keys(models)) expect(key).toMatch(/^gemini-2\.5-/)
  })

  test("exclude drops plain keys and /regex/ matches", () => {
    const plain = modelsFor({ options: { exclude: ["gemini-2.5-pro"] } })
    expect(plain["gemini-2.5-pro"]).toBeUndefined()
    expect(plain["gemini-2.5-flash"]).toBeDefined()

    const regex = modelsFor({ options: { exclude: ["/^claude-/"] } })
    for (const key of Object.keys(regex)) expect(key.startsWith("claude-")).toBe(false)
  })

  test("regionVariants adds key@region for every region of each entry", () => {
    const models = modelsFor({ options: { regionVariants: true } })
    expect(models["gemini-2.5-flash"]).toBeDefined()
    expect(models["gemini-2.5-flash@us-east5"].id).toBe("gemini-2.5-flash@us-east5")
    expect(models["gemini-2.5-flash@global"].api.id).toBe("gemini-2.5-flash")
    const expected = CATALOG.reduce((n, e) => n + 1 + e.regions.length, 0)
    expect(Object.keys(models).length).toBe(expected)
  })

  test("include and exclude combine with regionVariants", () => {
    const models = modelsFor({ options: { include: ["/^gemini-2\\.5-flash/"], exclude: ["gemini-2.5-flash-tts"], regionVariants: true } })
    expect(models["gemini-2.5-flash-tts@global"]).toBeUndefined()
    expect(models["gemini-2.5-flash@us-east5"]).toBeDefined()
    expect(Object.keys(models).every((k) => k.startsWith("gemini-2.5-flash"))).toBe(true)
  })
})

describe("toModel", () => {
  test("maps a reasoning text model to the OpenCode shape", () => {
    const model = toModel(lookup("gemini-2.5-flash").entry)
    expect(model.id).toBe("gemini-2.5-flash")
    expect(model.name).toBe("Gemini 2.5 Flash")
    expect(model.api).toEqual({ id: "gemini-2.5-flash", url: BASE, npm: "@ai-sdk/google" })
    expect(model.limit).toEqual({ context: 1048576, output: 65536 })
    expect(model.capabilities.reasoning).toBe(true)
    expect(model.capabilities.toolcall).toBe(true)
    expect(model.capabilities.attachment).toBe(true)
    expect(model.capabilities.input.image).toBe(true)
    expect(model.capabilities.output).toEqual({ text: true, image: false, audio: false, video: false })
    expect(model.cost).toEqual({ input: 0, output: 0, cache: { read: 0, write: 0 } })
    expect(model.status).toBe("active")
    expect(model.variants).toEqual({ low: {}, medium: {}, high: {} })
  })

  test("marks preview and experimental stages as beta", () => {
    expect(toModel(lookup("gemini-3-flash-preview").entry).status).toBe("beta")
    expect(toModel(lookup("text-embedding-large-exp-03-07").entry).status).toBe("beta")
    expect(toModel(lookup("gemini-2.5-flash").entry).status).toBe("active")
  })

  test("non-reasoning models have no variants and no attachment for text-only input", () => {
    const model = toModel(lookup("text-embedding-005").entry)
    expect(model.variants).toBeUndefined()
    expect(model.capabilities.attachment).toBe(false)
    expect(model.capabilities.output).toEqual({ text: false, image: false, audio: false, video: false })
  })

  test("a pinned region changes the id and name, and keeps the api id", () => {
    const model = toModel(lookup("gemini-2.5-flash").entry, "us-east5")
    expect(model.id).toBe("gemini-2.5-flash@us-east5")
    expect(model.name).toBe("Gemini 2.5 Flash (us-east5)")
    expect(model.api.id).toBe("gemini-2.5-flash")
  })

  test("MaaS models send the publisher-qualified api id", () => {
    const model = toModel(lookup("deepseek-v3.2-maas").entry)
    expect(model.id).toBe("deepseek-v3.2-maas")
    expect(model.api).toEqual({ id: "deepseek-ai/deepseek-v3.2-maas", url: BASE, npm: "@ai-sdk/openai-compatible" })
  })
})

describe("configModels", () => {
  test("uses the OpenCode config shape, keyed by magpie key", () => {
    const models = configModels()
    expect(Object.keys(models).length).toBe(CATALOG.length)
    expect(models["claude-haiku-4-5"]).toEqual({
      name: "Claude Haiku 4.5",
      id: "claude-haiku-4-5@20251001",
      limit: { context: 200000, output: 64000 },
      reasoning: true,
      tool_call: true,
      modalities: { input: ["text", "image", "pdf"], output: ["text"] },
      provider: { npm: "@ai-sdk/anthropic", api: BASE },
      variants: { low: {}, medium: {}, high: {} },
    })
    expect(models["deepseek-v3.2-maas"].id).toBe("deepseek-ai/deepseek-v3.2-maas")
    expect(models["deepseek-v3.2-maas"].provider.npm).toBe("@ai-sdk/openai-compatible")
  })

  test("non-reasoning models omit variants", () => {
    expect(configModels()["text-embedding-005"].variants).toBeUndefined()
  })
})

describe("mergeDiscovered", () => {
  const model = (id, publisher, family, regions) => ({ id, publisher, family, regions })

  test("returns the static catalog when nothing was discovered", () => {
    expect(mergeDiscovered(null)).toBe(CATALOG)
  })

  test("uses discovered regions and keeps catalog metadata for known ids", () => {
    const [entry, ...rest] = mergeDiscovered({ models: [model("gemini-2.5-flash", "google", "gemini", ["us-east5", "global"])], unavailable: [] })
    expect(rest).toEqual([])
    expect(entry.key).toBe("gemini-2.5-flash")
    expect(entry.regions).toEqual(["global", "us-east5"])
    expect(entry.context).toBe(1048576)
  })

  test("only discovered ids are kept; unavailable ones are dropped", () => {
    const merged = mergeDiscovered({
      models: [model("gemini-2.5-flash", "google", "gemini", ["global"])],
      unavailable: [{ id: "gemini-2.5-pro", publisher: "google", family: "gemini", reason: "forbidden" }],
    })
    expect(merged.map((e) => e.key)).toEqual(["gemini-2.5-flash"])
  })

  test("unknown ids get family defaults", () => {
    const [chat] = mergeDiscovered({ models: [model("gemini-9-flash", "google", "gemini", ["global"])], unavailable: [] })
    expect(chat.key).toBe("gemini-9-flash")
    expect(chat.name).toBe("gemini-9-flash")
    expect(chat.context).toBe(131072)
    expect(chat.output).toBe(8192)
    expect(chat.toolCall).toBe(true)
    expect(chat.input).toContain("image")

    const [embed] = mergeDiscovered({ models: [model("text-embedding-999", "google", "embedding", ["us-central1"])], unavailable: [] })
    expect(embed.output).toBe(0)
    expect(embed.outputs).toEqual([])
  })

  test("an unknown maas id keeps its publisher for the api id", () => {
    const [entry] = mergeDiscovered({ models: [model("new-model-maas", "qwen", "maas", ["global"])], unavailable: [] })
    expect(apiIdOf(entry)).toBe("qwen/new-model-maas")
  })

  test("skips discovered models whose family is unknown", () => {
    expect(mergeDiscovered({ models: [model("mystery", "google", "mystery", ["global"])], unavailable: [] })).toEqual([])
  })

  test("two versions of one model share a key; the catalog's own id wins in either order", () => {
    const versioned = model("claude-haiku-5-5@20261005", "anthropic", "claude", ["global"])
    const plain = model("claude-haiku-5-5", "anthropic", "claude", ["global", "us-central1"])
    for (const models of [[versioned, plain], [plain, versioned]]) {
      const merged = mergeDiscovered({ models, unavailable: [] })
      expect(merged.length).toBe(1)
      expect(merged[0].id).toBe("claude-haiku-5-5")
      expect(merged[0].regions).toEqual(["global", "us-central1"])
    }
  })
})

describe("orderRegions", () => {
  test("puts global first, then us-central1, then the rest in order", () => {
    expect(orderRegions(["us-east5", "us-central1", "global", "europe-west1"])).toEqual(["global", "us-central1", "us-east5", "europe-west1"])
  })
})

describe("loadDiscovered", () => {
  test("returns null when there is no cache file, and for an unreadable one", async () => {
    const dir = await mkdtemp(join(tmpdir(), "vertex-catalog-"))
    try {
      expect(await loadDiscovered(dir, "p1")).toBeNull()
      await mkdir(join(dir, "vertex-ai"), { recursive: true })
      await writeFile(discoveredFile(dir, "p1"), "{not json")
      expect(await loadDiscovered(dir, "p1")).toBeNull()
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("reads the file that discover writes", async () => {
    const dir = await mkdtemp(join(tmpdir(), "vertex-catalog-"))
    try {
      const data = { project: "p2", discoveredAt: "2026-10-14T00:00:00.000Z", models: [{ id: "gemini-2.5-flash", publisher: "google", family: "gemini", regions: ["global"] }], unavailable: [] }
      await mkdir(join(dir, "vertex-ai"), { recursive: true })
      await writeFile(discoveredFile(dir, "p2"), JSON.stringify(data))
      expect(await loadDiscovered(dir, "p2")).toEqual(data)
      expect(discoveredFile(dir, "p2")).toBe(join(dir, "vertex-ai", "catalog-p2.json"))
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })
})
