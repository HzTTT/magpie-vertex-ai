import { afterAll, beforeAll, describe, expect, test } from "bun:test"
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { handle } from "../lib/handlers/gemini.mjs"

const GEMINI = { id: "gemini-2.5-flash", key: "gemini-2.5-flash", publisher: "google", family: "gemini", regions: ["global", "us-central1", "europe-west4"] }
const IMAGE = { id: "gemini-2.5-flash-image", key: "gemini-2.5-flash-image", publisher: "google", family: "gemini-image", regions: ["us-central1"] }

let mediaDir

beforeAll(async () => {
  mediaDir = await mkdtemp(join(tmpdir(), "vertex-gemini-"))
})

afterAll(async () => {
  await rm(mediaDir, { recursive: true, force: true })
})

function json(status, value) {
  return new Response(JSON.stringify(value), { status, headers: { "content-type": "application/json" } })
}

function chunkedSse(parts) {
  const enc = new TextEncoder()
  const stream = new ReadableStream({
    start(controller) {
      for (const part of parts) controller.enqueue(enc.encode(part))
      controller.close()
    },
  })
  return new Response(stream, { status: 200, headers: { "content-type": "text/event-stream" } })
}

function fakeCtx(respond, overrides = {}) {
  const calls = []
  const logs = []
  const ctx = {
    project: "my-project",
    location: "auto",
    token: async () => "tok",
    fetch: async (url, init) => {
      calls.push({ url, init })
      return respond(calls.length - 1, url, init)
    },
    mediaDir,
    options: {},
    log: (msg) => logs.push(msg),
    ...overrides,
  }
  return { ctx, calls, logs }
}

function request(entry, body, headers = {}) {
  return {
    protocol: "gemini",
    suffix: `/models/${entry.id}:streamGenerateContent?alt=sse`,
    entry,
    pinned: undefined,
    apiId: entry.id,
    body,
    raw: undefined,
    headers: new Headers(headers),
    stream: true,
    signal: undefined,
  }
}

describe("gemini handler", () => {
  test("posts to the regional streamGenerateContent URL with Bearer auth and no model field", async () => {
    const { ctx, calls } = fakeCtx(() => chunkedSse(['data: {"candidates":[]}\n\n']), { location: "us-central1" })
    const res = await handle(request(GEMINI, { model: "gemini-2.5-flash", contents: [] }, { "x-goog-api-key": "SECRET", authorization: "Bearer SECRET" }), ctx)
    expect(res.status).toBe(200)
    expect(calls).toHaveLength(1)
    expect(calls[0].url).toBe(
      "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-2.5-flash:streamGenerateContent?alt=sse",
    )
    const headers = new Headers(calls[0].init.headers)
    expect(headers.get("authorization")).toBe("Bearer tok")
    expect(headers.get("x-goog-api-key")).toBeNull()
    expect(JSON.parse(calls[0].init.body)).toEqual({ contents: [] })
  })

  test("moves to the next region on 429 and streams the first success", async () => {
    const sse = 'data: {"candidates":[]}\n\n'
    const { ctx, calls } = fakeCtx((i) =>
      i === 0
        ? json(429, { error: { code: 429, message: "quota", status: "RESOURCE_EXHAUSTED" } })
        : chunkedSse([sse]),
    )
    const res = await handle(request(GEMINI, { contents: [] }), ctx)
    expect(res.status).toBe(200)
    expect(calls.map((c) => new URL(c.url).host)).toEqual(["aiplatform.googleapis.com", "us-central1-aiplatform.googleapis.com"])
    expect(await res.text()).toBe(sse)
  })

  test("returns a non-retryable upstream error from the first region without trying others", async () => {
    const body = '{"error":{"code":400,"message":"Invalid JSON payload","status":"INVALID_ARGUMENT"}}'
    const { ctx, calls } = fakeCtx(() => new Response(body, { status: 400, headers: { "content-type": "application/json" } }))
    const res = await handle(request(GEMINI, { contents: [] }), ctx)
    expect(res.status).toBe(400)
    expect(calls).toHaveLength(1)
    expect(await res.text()).toBe(body)
  })

  test("wraps a non-JSON final failure in the Gemini error shape", async () => {
    const { ctx } = fakeCtx(() => new Response("<html>Not Found</html>", { status: 404, headers: { "content-type": "text/html" } }))
    const res = await handle(request(GEMINI, { contents: [] }), ctx)
    expect(res.status).toBe(404)
    const body = await res.json()
    expect(body.error.message).toContain("Not Found")
  })

  test("drops AI Studio cachedContent ids and keeps Vertex cache resource names", async () => {
    const { ctx, calls } = fakeCtx(() => chunkedSse([""]))
    await handle(request(GEMINI, { contents: [], cachedContent: "cachedContents/abc" }), ctx)
    await handle(request(GEMINI, { contents: [], cachedContent: "projects/my-project/locations/us-central1/cachedContents/9" }), ctx)
    expect(JSON.parse(calls[0].init.body)).toEqual({ contents: [] })
    expect(JSON.parse(calls[1].init.body).cachedContent).toBe("projects/my-project/locations/us-central1/cachedContents/9")
  })

  test("text models keep tools and get no response modalities", async () => {
    const tools = [{ functionDeclarations: [{ name: "f" }] }]
    const { ctx, calls } = fakeCtx(() => chunkedSse([""]))
    await handle(request(GEMINI, { contents: [], tools }), ctx)
    expect(JSON.parse(calls[0].init.body)).toEqual({ contents: [], tools })
  })

  test("image models request TEXT and IMAGE output and lose tools and thinking config", async () => {
    const { ctx, calls } = fakeCtx(() => chunkedSse([""]), { location: "us-central1" })
    await handle(
      request(IMAGE, {
        contents: [],
        tools: [{ functionDeclarations: [{ name: "f" }] }],
        toolConfig: { functionCallingConfig: { mode: "AUTO" } },
        generationConfig: { temperature: 0.2, thinkingConfig: { includeThoughts: true } },
      }),
      ctx,
    )
    expect(calls[0].url).toContain("/models/gemini-2.5-flash-image:streamGenerateContent?alt=sse")
    expect(JSON.parse(calls[0].init.body)).toEqual({
      contents: [],
      generationConfig: { temperature: 0.2, responseModalities: ["TEXT", "IMAGE"] },
    })
  })

  test("image models keep response modalities the request already sets", async () => {
    const { ctx, calls } = fakeCtx(() => chunkedSse([""]))
    await handle(request(IMAGE, { contents: [], generationConfig: { responseModalities: ["IMAGE"] } }), ctx)
    expect(JSON.parse(calls[0].init.body).generationConfig.responseModalities).toEqual(["IMAGE"])
  })

  test("saves generated images, adds a link part after each, and passes other lines through unchanged", async () => {
    const png = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
    const b64 = Buffer.from(png).toString("base64")
    const text = 'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Here"}]}}]}\n'
    const keepalive = ": keepalive\n"
    const blank = "\n"
    const image = `data: {"candidates":[{"content":{"role":"model","parts":[{"inlineData":{"mimeType":"image/png","data":"${b64}"}}]},"finishReason":"STOP"}]}\r\n`
    const upstream = text + keepalive + blank + image + blank
    // The cut lands inside the image line, so the transformer must buffer a partial data: line.
    const cut = text.length + keepalive.length + blank.length + 40
    const { ctx } = fakeCtx(() => chunkedSse([upstream.slice(0, cut), upstream.slice(cut)]))
    const res = await handle(request(IMAGE, { contents: [] }), ctx)
    const lines = (await res.text()).split("\n")
    expect(lines[0]).toBe(text.slice(0, -1))
    expect(lines[1]).toBe(": keepalive")
    expect(lines[2]).toBe("")
    expect(lines[3].endsWith("\r")).toBe(true)
    const chunk = JSON.parse(lines[3].replace(/^data: /, "").trimEnd())
    const [inline, note] = chunk.candidates[0].content.parts
    expect(inline).toEqual({ inlineData: { mimeType: "image/png", data: b64 } })
    expect(note.text).toMatch(/^\n!\[image\]\(file:\/\/\S+\)\n\nSaved to: \S+\n$/)
    const path = note.text.match(/Saved to: (\S+)\n$/)[1]
    expect(path.startsWith(mediaDir)).toBe(true)
    expect(new Uint8Array(await readFile(path))).toEqual(png)
    expect(chunk.candidates[0].finishReason).toBe("STOP")
    expect(lines.slice(4)).toEqual(["", ""])
  })

  test("labels generated audio as an [audio] link", async () => {
    const data = Buffer.from("RIFF").toString("base64")
    const line = `data: {"candidates":[{"content":{"role":"model","parts":[{"inlineData":{"mimeType":"audio/wav","data":"${data}"}}]}}]}\n`
    const { ctx } = fakeCtx(() => chunkedSse([line]))
    const res = await handle(request(GEMINI, { contents: [] }), ctx)
    const chunk = JSON.parse((await res.text()).replace(/^data: /, "").trim())
    expect(chunk.candidates[0].content.parts[1].text).toMatch(/^\n\[audio\]\(file:\/\/\S+\)\n\nSaved to: \S+\n$/)
  })

  test("keeps the original line and logs when saving media fails", async () => {
    const blocker = join(mediaDir, "blocked-file")
    await writeFile(blocker, "not a directory")
    const data = Buffer.from([1, 2, 3]).toString("base64")
    const line = `data: {"candidates":[{"content":{"role":"model","parts":[{"inlineData":{"mimeType":"image/png","data":"${data}"}}]}}]}\n`
    const { ctx, logs } = fakeCtx(() => chunkedSse([line]), { mediaDir: join(blocker, "out") })
    const res = await handle(request(IMAGE, { contents: [] }), ctx)
    expect(await res.text()).toBe(line)
    expect(logs).toHaveLength(1)
  })
})
