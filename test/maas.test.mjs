import { describe, expect, test } from "bun:test"
import { handle } from "../lib/handlers/maas.mjs"

const DEEPSEEK = { id: "deepseek-v3.2-maas", key: "deepseek-v3.2-maas", publisher: "deepseek-ai", family: "maas", regions: ["global"] }
const QWEN = { id: "qwen3-235b-a22b-instruct-2507-maas", key: "qwen3-235b-a22b-instruct-2507-maas", publisher: "qwen", family: "maas", regions: ["us-central1", "global"] }

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
  const ctx = {
    project: "my-project",
    location: "auto",
    token: async () => "tok",
    fetch: async (url, init) => {
      calls.push({ url, init })
      return respond(calls.length - 1, url, init)
    },
    mediaDir: "/tmp/unused",
    options: {},
    log: () => {},
    ...overrides,
  }
  return { ctx, calls }
}

function request(entry, body, headers = {}) {
  return {
    protocol: "openai",
    suffix: "/chat/completions",
    entry,
    pinned: undefined,
    apiId: `${entry.publisher}/${entry.id}`,
    body,
    raw: undefined,
    headers: new Headers(headers),
    stream: body.stream === true,
    signal: undefined,
  }
}

const GLOBAL_URL = "https://aiplatform.googleapis.com/v1beta1/projects/my-project/locations/global/endpoints/openapi/chat/completions"

describe("maas handler", () => {
  test("posts to the global OpenAI-compatible endpoint with the publisher-qualified model id", async () => {
    const { ctx, calls } = fakeCtx(() => json(200, { choices: [] }))
    const res = await handle(
      request(
        DEEPSEEK,
        {
          model: "deepseek-v3.2-maas",
          messages: [{ role: "user", content: "hi" }],
          temperature: 0.2,
          store: false,
          parallel_tool_calls: true,
          reasoning_effort: "low",
        },
        { "x-api-key": "LEAK" },
      ),
      ctx,
    )
    expect(res.status).toBe(200)
    expect(calls[0].url).toBe(GLOBAL_URL)
    const headers = new Headers(calls[0].init.headers)
    expect(headers.get("authorization")).toBe("Bearer tok")
    expect(headers.get("x-api-key")).toBeNull()
    expect(JSON.parse(calls[0].init.body)).toEqual({
      model: "deepseek-ai/deepseek-v3.2-maas",
      messages: [{ role: "user", content: "hi" }],
      temperature: 0.2,
      store: false,
      parallel_tool_calls: true,
      reasoning_effort: "low",
    })
  })

  test("adds stream_options include_usage to streaming requests unless the client set them", async () => {
    const { ctx, calls } = fakeCtx(() => chunkedSse([""]))
    await handle(request(DEEPSEEK, { messages: [], stream: true }), ctx)
    await handle(request(DEEPSEEK, { messages: [], stream: true, stream_options: { include_usage: false } }), ctx)
    expect(JSON.parse(calls[0].init.body).stream_options).toEqual({ include_usage: true })
    expect(JSON.parse(calls[1].init.body).stream_options).toEqual({ include_usage: false })
  })

  test("adds no stream_options to non-streaming requests", async () => {
    const { ctx, calls } = fakeCtx(() => json(200, {}))
    await handle(request(DEEPSEEK, { messages: [], stream: false }), ctx)
    expect(JSON.parse(calls[0].init.body)).not.toHaveProperty("stream_options")
  })

  test("retries the next region on 'only available via global endpoint'", async () => {
    const { ctx, calls } = fakeCtx((i) =>
      i === 0
        ? json(400, { error: { code: 400, message: "This model is only available via global endpoint.", status: "INVALID_ARGUMENT" } })
        : json(200, { choices: [] }),
    )
    const res = await handle(request(QWEN, { messages: [] }), { ...ctx, location: "us-central1" })
    expect(res.status).toBe(200)
    expect(calls.map((c) => new URL(c.url).host)).toEqual(["us-central1-aiplatform.googleapis.com", "aiplatform.googleapis.com"])
    expect(calls[1].url).toBe(GLOBAL_URL)
  })

  test("converts a Google-shaped error to the OpenAI error shape", async () => {
    const message = "Invalid JSON payload received."
    const { ctx } = fakeCtx(() => json(400, { error: { code: 400, message, status: "INVALID_ARGUMENT" } }))
    const res = await handle(request(DEEPSEEK, { messages: [] }), ctx)
    expect(res.status).toBe(400)
    const body = await res.json()
    expect(body.error.message).toBe(message)
  })

  test("passes an OpenAI-shaped upstream error through unchanged", async () => {
    const text = '{"error":{"message":"bad","type":"invalid_request_error","param":null,"code":null}}'
    const { ctx } = fakeCtx(() => new Response(text, { status: 400, headers: { "content-type": "application/json" } }))
    const res = await handle(request(DEEPSEEK, { messages: [] }), ctx)
    expect(await res.text()).toBe(text)
  })

  test("passes streamed chunks through byte for byte", async () => {
    const sse = 'data: {"choices":[{"delta":{"content":"OK"}}]}\n\ndata: [DONE]\n\n'
    const { ctx } = fakeCtx(() => chunkedSse([sse.slice(0, 30), sse.slice(30)]))
    const res = await handle(request(DEEPSEEK, { messages: [], stream: true }), ctx)
    expect(res.headers.get("content-type")).toBe("text/event-stream")
    expect(await res.text()).toBe(sse)
  })
})
