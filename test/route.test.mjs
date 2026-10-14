import { describe, expect, test } from "bun:test"
import { BASE } from "../lib/regions.mjs"
import { route } from "../lib/route.mjs"

const NAMES = ["gemini", "tts", "veo", "lyria", "claude", "maas", "embeddings", "transcribe"]
const ctx = { project: "demo", location: "auto", log: () => {} }

// A small catalog covering every family; ids and publishers match apiIdOf.
const entries = [
  { id: "gemini-2.5-flash", key: "gemini-2.5-flash", publisher: "google", family: "gemini", regions: ["global", "us-central1"] },
  { id: "gemini-2.5-flash-image", key: "gemini-2.5-flash-image", publisher: "google", family: "gemini-image", regions: ["us-central1"] },
  { id: "gemini-2.5-flash-tts", key: "gemini-2.5-flash-tts", publisher: "google", family: "gemini-tts", regions: ["global"] },
  { id: "chirp-3", key: "chirp-3", publisher: "google", family: "chirp-tts", regions: ["us-central1"] },
  { id: "veo-3.1-generate-001", key: "veo-3.1-generate-001", publisher: "google", family: "veo", regions: ["us-central1"] },
  { id: "lyria-002", key: "lyria-002", publisher: "google", family: "lyria", regions: ["us-central1"] },
  { id: "claude-opus-5", key: "claude-opus-5", publisher: "anthropic", family: "claude", regions: ["global"] },
  { id: "deepseek-v3.2-maas", key: "deepseek-v3.2-maas", publisher: "deepseek-ai", family: "maas", regions: ["global"] },
  { id: "text-embedding-005", key: "text-embedding-005", publisher: "google", family: "embedding", regions: ["us-central1"] },
]

// harness returns a route runner whose handlers record what they were given.
function harness() {
  const calls = []
  const handlers = Object.fromEntries(
    NAMES.map((name) => [
      name,
      {
        handle: async (req, c) => {
          calls.push({ name, req, ctx: c })
          return new Response(name)
        },
      },
    ]),
  )
  const run = (path, init = {}) => route(`${BASE}${path}`, { method: "POST", ...init }, ctx, { entries, handlers })
  const json = (path, body, headers = {}) => run(path, { headers: { "content-type": "application/json", ...headers }, body: JSON.stringify(body) })
  return { calls, run, json }
}

async function multipart(fields) {
  const form = new FormData()
  for (const [k, v] of Object.entries(fields)) form.set(k, v)
  form.set("file", new Blob([new Uint8Array([82, 73, 70, 70])], { type: "audio/wav" }), "a.wav")
  const built = new Response(form)
  return { contentType: built.headers.get("content-type"), body: await built.arrayBuffer() }
}

describe("route dispatch", () => {
  test.each([
    ["/models/gemini-2.5-flash:streamGenerateContent?alt=sse", "gemini"],
    ["/models/gemini-2.5-flash-image:generateContent", "gemini"],
    ["/models/gemini-2.5-flash-tts:generateContent", "tts"],
    ["/models/chirp-3:synthesize", "tts"],
    ["/models/veo-3.1-generate-001:predictLongRunning", "veo"],
    ["/models/lyria-002:predict", "lyria"],
  ])("%s goes to the %s handler", async (path, name) => {
    const h = harness()
    const res = await h.run(path, { headers: { "content-type": "application/json" }, body: JSON.stringify({ contents: [] }) })
    expect(await res.text()).toBe(name)
    expect(h.calls.map((c) => c.name)).toEqual([name])
  })

  test("a Gemini stream request carries the suffix, the parsed body and stream=true", async () => {
    const h = harness()
    await h.run("/models/gemini-2.5-flash:streamGenerateContent?alt=sse", {
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ contents: [{ role: "user", parts: [{ text: "hi" }] }] }),
    })
    const { req, ctx: seen } = h.calls[0]
    expect(req.protocol).toBe("gemini")
    expect(req.suffix).toBe("/models/gemini-2.5-flash:streamGenerateContent?alt=sse")
    expect(req.apiId).toBe("gemini-2.5-flash")
    expect(req.entry.family).toBe("gemini")
    expect(req.stream).toBe(true)
    expect(req.body.contents[0].parts[0].text).toBe("hi")
    expect(req.form).toBeNull()
    expect(seen).toBe(ctx)
  })

  test("a non-streaming Gemini request is not a stream", async () => {
    const h = harness()
    await h.run("/models/gemini-2.5-flash:generateContent", { headers: { "content-type": "application/json" }, body: "{}" })
    expect(h.calls[0].req.stream).toBe(false)
  })

  test("a region-pinned model id carries the region to the handler", async () => {
    const h = harness()
    await h.run("/models/gemini-2.5-flash@us-central1:generateContent", { headers: { "content-type": "application/json" }, body: "{}" })
    expect(h.calls[0].req.pinned).toBe("us-central1")
    expect(h.calls[0].req.entry.id).toBe("gemini-2.5-flash")
  })

  test("/messages goes to the claude handler, keeps anthropic-beta and drops key headers", async () => {
    const h = harness()
    const res = await h.json(
      "/messages",
      { model: "claude-opus-5", stream: true, messages: [] },
      { "anthropic-beta": "tools-2025", authorization: "Bearer secret", "x-goog-api-key": "secret", "x-api-key": "secret" },
    )
    expect(await res.text()).toBe("claude")
    const { req } = h.calls[0]
    expect(req.protocol).toBe("anthropic")
    expect(req.apiId).toBe("claude-opus-5")
    expect(req.stream).toBe(true)
    expect(req.headers.get("anthropic-beta")).toBe("tools-2025")
    expect(req.headers.get("authorization")).toBeNull()
    expect(req.headers.get("x-goog-api-key")).toBeNull()
    expect(req.headers.get("x-api-key")).toBeNull()
  })

  test("/chat/completions goes to the maas handler under its publisher-qualified id", async () => {
    const h = harness()
    const res = await h.json("/chat/completions", { model: "deepseek-ai/deepseek-v3.2-maas", messages: [] })
    expect(await res.text()).toBe("maas")
    expect(h.calls[0].req.protocol).toBe("openai")
    expect(h.calls[0].req.apiId).toBe("deepseek-ai/deepseek-v3.2-maas")
    expect(h.calls[0].req.stream).toBe(false)
  })

  test("/chat/completions with a Gemini chat model goes to the OpenAI-compatible (maas) handler", async () => {
    const h = harness()
    const res = await h.json("/chat/completions", { model: "gemini-2.5-flash", messages: [] })
    expect(await res.text()).toBe("maas")
    expect(h.calls[0].req.entry.id).toBe("gemini-2.5-flash")
  })

  test("/embeddings goes to the embeddings handler", async () => {
    const h = harness()
    const res = await h.json("/embeddings", { model: "text-embedding-005", input: "x" })
    expect(await res.text()).toBe("embeddings")
    expect(h.calls[0].req.entry.family).toBe("embedding")
  })

  test.each(["/audio/transcriptions", "/audio/translations"])("%s is multipart and routes by the form's model field", async (path) => {
    const h = harness()
    const mp = await multipart({ model: "gemini-2.5-flash" })
    const res = await h.run(path, { headers: { "content-type": mp.contentType }, body: mp.body })
    expect(await res.text()).toBe("transcribe")
    const { req } = h.calls[0]
    expect(req.protocol).toBe("openai")
    expect(req.body).toBeNull()
    expect(req.form.get("model")).toBe("gemini-2.5-flash")
    expect(req.form.get("file").name).toBe("a.wav")
  })
})

describe("route errors", () => {
  test("an unknown model is a 404 in the caller's protocol", async () => {
    const h = harness()
    let res = await h.json("/chat/completions", { model: "nope" })
    expect(res.status).toBe(404)
    expect((await res.json()).error.message).toContain("nope")

    res = await h.json("/models/nope:generateContent", { contents: [] })
    expect(res.status).toBe(404)
    const gemini = (await res.json()).error
    expect(gemini.code).toBe(404)
    expect(gemini.status).toBe("NOT_FOUND")

    res = await h.json("/messages", { model: "nope" })
    expect(res.status).toBe(404)
    const anthropic = await res.json()
    expect(anthropic.type).toBe("error")
    expect(anthropic.error.type).toBe("not_found_error")

    expect(h.calls.length).toBe(0)
  })

  test.each([
    ["/chat/completions", "openai", { model: "gemini-2.5-flash-tts" }],
    ["/models/text-embedding-005:predict", "gemini", { instances: [] }],
    ["/embeddings", "openai", { model: "deepseek-ai/deepseek-v3.2-maas" }],
    ["/messages", "anthropic", { model: "gemini-2.5-flash" }],
    ["/audio/transcriptions", "openai", { model: "claude-opus-5" }],
  ])("%s with the wrong family is a 400 in the %s shape, and nothing is dispatched", async (path, protocol, body) => {
    const h = harness()
    const res = await h.json(path, body)
    expect(res.status).toBe(400)
    const parsed = await res.json()
    if (protocol === "gemini") expect(parsed.error.code).toBe(400)
    else if (protocol === "anthropic") expect(parsed.error.type).toBe("invalid_request_error")
    else expect(parsed.error.message).toEqual(expect.any(String))
    expect(h.calls.length).toBe(0)
  })

  test("a family mismatch names the endpoint the model is served on", async () => {
    const h = harness()
    const res = await h.json("/chat/completions", { model: "gemini-2.5-flash-tts" })
    expect((await res.json()).error.message).toContain("/models/")
  })

  test("an endpoint magpie does not route to a plugin is a 404", async () => {
    const h = harness()
    const res = await h.json("/responses", { model: "gemini-2.5-flash" })
    expect(res.status).toBe(404)
    expect(h.calls.length).toBe(0)
  })

  test("a body that is not JSON is a 400", async () => {
    const h = harness()
    const res = await h.run("/chat/completions", { headers: { "content-type": "application/json" }, body: "{" })
    expect(res.status).toBe(400)
    expect(h.calls.length).toBe(0)
  })

  test("an ordinary handler error becomes a 502 in the caller's protocol", async () => {
    const res = await route(
      `${BASE}/chat/completions`,
      { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ model: "deepseek-ai/deepseek-v3.2-maas" }) },
      ctx,
      {
        entries,
        handlers: {
          maas: {
            handle: async () => {
              throw new Error("upstream exploded")
            },
          },
        },
      },
    )
    expect(res.status).toBe(502)
    expect((await res.json()).error.message).toContain("upstream exploded")
  })

  test("a sign-in error from a handler reaches the host", async () => {
    const err = Object.assign(new Error("refused"), { signIn: "expired" })
    await expect(
      route(
        `${BASE}/messages`,
        { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ model: "claude-opus-5" }) },
        ctx,
        {
          entries,
          handlers: {
            claude: {
              handle: async () => {
                throw err
              },
            },
          },
        },
      ),
    ).rejects.toBe(err)
  })
})
