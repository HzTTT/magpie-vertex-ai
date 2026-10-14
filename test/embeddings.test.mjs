import { describe, expect, test } from "bun:test"
import { handle } from "../lib/handlers/embeddings.mjs"

const json = (obj, status = 200) =>
  new Response(JSON.stringify(obj), { status, headers: { "content-type": "application/json" } })

const fail = (status, message) => json({ error: { code: status, message, status: "FAILED" } }, status)

// Echoes each :predict instance as a vector whose first value is the input length.
const echoPredict = (_url, body) =>
  json({
    predictions: body.instances.map(i => ({
      embeddings: { values: [i.content.length, 0.25], statistics: { token_count: 3 } },
    })),
  })

function harness(respond) {
  const calls = []
  const ctx = {
    project: "my-project",
    location: "auto",
    token: async () => "test-token",
    fetch: async (url, init) => {
      const body = JSON.parse(init.body)
      calls.push({ url, method: init.method, auth: init.headers.authorization, body })
      return respond(url, body, calls.length)
    },
    mediaDir: "/tmp/unused",
    options: {},
    log: () => {},
  }
  return { ctx, calls }
}

const entry = (id, regions = ["us-central1"]) => ({ id, key: id, publisher: "google", family: "embedding", regions })

const request = (e, body) => ({
  protocol: "openai",
  suffix: "/embeddings",
  entry: e,
  pinned: undefined,
  apiId: e.id,
  body,
  raw: null,
  headers: new Headers(),
  stream: false,
  signal: undefined,
})

describe("text embedding models (:predict)", () => {
  test("maps one string to a single :predict call and translates the reply", async () => {
    const { ctx, calls } = harness(echoPredict)
    const res = await handle(request(entry("text-embedding-005"), { model: "text-embedding-005", input: "hello" }), ctx)

    expect(res.status).toBe(200)
    expect(calls).toHaveLength(1)
    expect(calls[0].url).toBe(
      "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/text-embedding-005:predict",
    )
    expect(calls[0].method).toBe("POST")
    expect(calls[0].auth).toBe("Bearer test-token")
    expect(calls[0].body).toEqual({ instances: [{ content: "hello" }], parameters: { autoTruncate: true } })
    expect(await res.json()).toEqual({
      object: "list",
      data: [{ object: "embedding", index: 0, embedding: [5, 0.25] }],
      model: "text-embedding-005",
      usage: { prompt_tokens: 3, total_tokens: 3 },
    })
  })

  test("splits 600 inputs into 250/250/100 instance batches and keeps input order", async () => {
    const { ctx, calls } = harness(echoPredict)
    const inputs = Array.from({ length: 600 }, (_, k) => "x".repeat(k + 1))
    const res = await handle(request(entry("text-multilingual-embedding-002"), { input: inputs }), ctx)

    expect(calls.map(c => c.body.instances.length).sort((a, b) => b - a)).toEqual([250, 250, 100])
    const body = await res.json()
    expect(body.data).toHaveLength(600)
    expect(body.data.map(d => d.index)).toEqual(inputs.map((_, k) => k))
    expect(body.data.every((d, k) => d.embedding[0] === k + 1)).toBe(true)
    expect(body.usage).toEqual({ prompt_tokens: 1800, total_tokens: 1800 })
  })

  test("forwards dimensions as outputDimensionality and task_type on each instance", async () => {
    const { ctx, calls } = harness(echoPredict)
    await handle(
      request(entry("text-embedding-005"), { input: ["q"], dimensions: 256, task_type: "RETRIEVAL_QUERY" }),
      ctx,
    )
    expect(calls[0].body).toEqual({
      instances: [{ content: "q", task_type: "RETRIEVAL_QUERY" }],
      parameters: { autoTruncate: true, outputDimensionality: 256 },
    })
  })

  test("accepts an array of strings with encoding_format float unchanged", async () => {
    const { ctx } = harness(echoPredict)
    const res = await handle(request(entry("text-embedding-005"), { input: ["a", "bb"], encoding_format: "float" }), ctx)
    const body = await res.json()
    expect(body.data.map(d => d.embedding)).toEqual([[1, 0.25], [2, 0.25]])
  })

  test("reports a 502 when the reply has fewer predictions than inputs", async () => {
    const { ctx } = harness(() => json({ predictions: [] }))
    const res = await handle(request(entry("text-embedding-005"), { input: ["a"] }), ctx)
    expect(res.status).toBe(502)
  })
})

describe("gemini-embedding-001 (one instance per :predict call)", () => {
  test("sends one instance per call and keeps order when replies arrive out of order", async () => {
    const { ctx, calls } = harness(async (url, body) => {
      const k = body.instances[0].content.length - 1
      await new Promise(r => setTimeout(r, (5 - k) * 5))
      return echoPredict(url, body)
    })
    const res = await handle(
      request(entry("gemini-embedding-001"), { input: ["a", "bb", "ccc", "dddd", "eeeee"] }),
      ctx,
    )

    expect(calls).toHaveLength(5)
    expect(calls.every(c => c.body.instances.length === 1)).toBe(true)
    expect(calls.every(c => c.url.endsWith("/models/gemini-embedding-001:predict"))).toBe(true)
    const body = await res.json()
    expect(body.data.map(d => d.embedding[0])).toEqual([1, 2, 3, 4, 5])
  })
})

describe("gemini-embedding-2 (:embedContent only)", () => {
  test("calls :embedContent once per input and reads embedding.values and usageMetadata", async () => {
    const { ctx, calls } = harness((url, body) =>
      json({
        embedding: { values: [body.content.parts[0].text.length, 1] },
        usageMetadata: { promptTokenCount: 4, totalTokenCount: 4 },
      }),
    )
    const res = await handle(
      request(entry("gemini-embedding-2", ["global"]), { input: ["ab", "cde"], dimensions: 512, task_type: "RETRIEVAL_DOCUMENT" }),
      ctx,
    )

    expect(calls).toHaveLength(2)
    expect(calls.every(c => c.url === "https://aiplatform.googleapis.com/v1/projects/my-project/locations/global/publishers/google/models/gemini-embedding-2:embedContent")).toBe(true)
    expect(calls.map(c => c.body).sort((a, b) => a.content.parts[0].text.localeCompare(b.content.parts[0].text))).toEqual([
      { content: { parts: [{ text: "ab" }] }, taskType: "RETRIEVAL_DOCUMENT", outputDimensionality: 512 },
      { content: { parts: [{ text: "cde" }] }, taskType: "RETRIEVAL_DOCUMENT", outputDimensionality: 512 },
    ])
    const body = await res.json()
    expect(body.data.map(d => d.embedding)).toEqual([[2, 1], [3, 1]])
    expect(body.usage).toEqual({ prompt_tokens: 8, total_tokens: 8 })
  })
})

describe("multimodalembedding", () => {
  test("sends {instances:[{text}]} per input and reads predictions[i].textEmbedding", async () => {
    const { ctx, calls } = harness((_url, body) =>
      json({ predictions: body.instances.map(i => ({ textEmbedding: [i.text.length, 2] })) }),
    )
    const res = await handle(request(entry("multimodalembedding"), { input: ["hi", "there"] }), ctx)

    expect(calls).toHaveLength(2)
    const texts = calls.map(c => c.body.instances[0].text).sort()
    expect(texts).toEqual(["hi", "there"])
    expect(calls.every(c => c.body.parameters === undefined)).toBe(true)
    const body = await res.json()
    expect(body.data.map(d => d.embedding)).toEqual([[2, 2], [5, 2]])
    expect(body.usage).toEqual({ prompt_tokens: 0, total_tokens: 0 })
  })

  test("maps dimensions to parameters.dimension", async () => {
    const { ctx, calls } = harness((_url, body) =>
      json({ predictions: body.instances.map(() => ({ textEmbedding: [1] })) }),
    )
    await handle(request(entry("multimodalembedding"), { input: "hi", dimensions: 512 }), ctx)
    expect(calls[0].body).toEqual({ instances: [{ text: "hi" }], parameters: { dimension: 512 } })
  })
})

describe("encoding and validation", () => {
  test("encodes base64 as little-endian float32 bytes", async () => {
    const { ctx } = harness(() => json({ predictions: [{ embeddings: { values: [0.5, -1.25, 3], statistics: { token_count: 1 } } }] }))
    const res = await handle(request(entry("text-embedding-005"), { input: "x", encoding_format: "base64" }), ctx)
    const { data } = await res.json()

    expect(typeof data[0].embedding).toBe("string")
    const bytes = Buffer.from(data[0].embedding, "base64")
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
    expect([0, 4, 8].map(o => view.getFloat32(o, true))).toEqual([0.5, -1.25, 3])
  })

  test("rejects token-array input with 400 and never calls Vertex", async () => {
    for (const input of [[[1, 2, 3]], [1, 2, 3], [[1], "text"]]) {
      const { ctx, calls } = harness(echoPredict)
      const res = await handle(request(entry("text-embedding-005"), { input }), ctx)
      expect(res.status).toBe(400)
      expect((await res.json()).error.message).toMatch(/token array/)
      expect(calls).toHaveLength(0)
    }
  })

  test("rejects an unknown encoding_format and a non-positive dimensions", async () => {
    const { ctx, calls } = harness(echoPredict)
    const bad = await handle(request(entry("text-embedding-005"), { input: "x", encoding_format: "hex" }), ctx)
    const zeroDims = await handle(request(entry("text-embedding-005"), { input: "x", dimensions: 0 }), ctx)
    expect(bad.status).toBe(400)
    expect(zeroDims.status).toBe(400)
    expect(calls).toHaveLength(0)
  })
})

describe("region failover and errors", () => {
  test("moves to the next region on a retryable error", async () => {
    const { ctx, calls } = harness((url, body) =>
      url.includes("/locations/us-central1/") ? fail(429, "RESOURCE_EXHAUSTED") : echoPredict(url, body),
    )
    const res = await handle(request(entry("text-embedding-005", ["us-central1", "europe-west4"]), { input: "hi" }), ctx)

    expect(calls.map(c => c.url.split("/locations/")[1].split("/")[0])).toEqual(["us-central1", "europe-west4"])
    expect(res.status).toBe(200)
  })

  test("returns the upstream message for a non-retryable failure without trying other regions", async () => {
    const { ctx, calls } = harness(() => fail(400, "Input is too long for this model."))
    const res = await handle(request(entry("text-embedding-005", ["us-central1", "europe-west4"]), { input: "hi" }), ctx)

    expect(res.status).toBe(400)
    expect((await res.json()).error.message).toBe("Input is too long for this model.")
    expect(calls).toHaveLength(1)
  })
})
