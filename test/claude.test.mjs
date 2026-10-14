import { describe, expect, test } from "bun:test"
import { handle } from "../lib/handlers/claude.mjs"

const HAIKU = { id: "claude-haiku-4-5@20251001", key: "claude-haiku-4-5", publisher: "anthropic", family: "claude", regions: ["global", "us-central1"] }

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
    protocol: "anthropic",
    suffix: "/messages",
    entry,
    pinned: undefined,
    apiId: entry.id,
    body,
    raw: undefined,
    headers: new Headers(headers),
    stream: body.stream === true,
    signal: undefined,
  }
}

describe("claude handler", () => {
  test("rewrites the Anthropic body for rawPredict and moves betas from the header into the body", async () => {
    const { ctx, calls } = fakeCtx(() => json(200, { id: "msg_1", type: "message" }))
    const body = {
      model: "claude-haiku-4-5@20251001",
      max_tokens: 64,
      stream: false,
      messages: [{ role: "user", content: "hi" }],
      cache_control: { type: "ephemeral" },
      metadata: { user_id: "u1" },
      thinking: { type: "adaptive" },
      output_config: { effort: "low" },
      anthropic_beta: ["tools-2024-05-16"],
      tools: [
        { type: "web_search_20250305", name: "web_search", max_uses: 5 },
        { name: "get_weather", input_schema: { type: "object" } },
      ],
      tool_choice: { type: "auto" },
    }
    const res = await handle(
      request(HAIKU, body, { "anthropic-beta": "prompt-caching-2024-07-31, tools-2024-05-16", "x-api-key": "LEAK" }),
      ctx,
    )
    expect(res.status).toBe(200)
    expect(calls[0].url).toBe(
      "https://aiplatform.googleapis.com/v1/projects/my-project/locations/global/publishers/anthropic/models/claude-haiku-4-5@20251001:rawPredict",
    )
    const headers = new Headers(calls[0].init.headers)
    expect(headers.get("authorization")).toBe("Bearer tok")
    expect(headers.get("anthropic-beta")).toBeNull()
    expect(headers.get("x-api-key")).toBeNull()
    expect(JSON.parse(calls[0].init.body)).toEqual({
      max_tokens: 64,
      stream: false,
      messages: [{ role: "user", content: "hi" }],
      cache_control: { type: "ephemeral" },
      metadata: { user_id: "u1" },
      thinking: { type: "adaptive" },
      output_config: { effort: "low" },
      anthropic_version: "vertex-2023-10-16",
      anthropic_beta: ["tools-2024-05-16", "prompt-caching-2024-07-31"],
      tools: [{ name: "get_weather", input_schema: { type: "object" } }],
      tool_choice: { type: "auto" },
    })
  })

  test("streams through streamRawPredict and passes the SSE bytes through unchanged", async () => {
    const sse = 'event: message_start\ndata: {"type":"message_start"}\n\n'
    const { ctx, calls } = fakeCtx(() => chunkedSse([sse.slice(0, 20), sse.slice(20)]))
    const res = await handle(request(HAIKU, { max_tokens: 5, stream: true, messages: [] }), ctx)
    expect(calls[0].url).toMatch(/:streamRawPredict$/)
    expect(res.headers.get("content-type")).toBe("text/event-stream")
    expect(await res.text()).toBe(sse)
  })

  test("drops tool_choice together with a tools list that only held the injected web search", async () => {
    const { ctx, calls } = fakeCtx(() => json(200, {}))
    await handle(
      request(HAIKU, {
        max_tokens: 5,
        messages: [],
        tools: [{ type: "web_search_20250305", name: "web_search" }],
        tool_choice: { type: "auto" },
      }),
      ctx,
    )
    const sent = JSON.parse(calls[0].init.body)
    expect(sent.tools).toBeUndefined()
    expect(sent.tool_choice).toBeUndefined()
  })

  test("keeps the @version suffix of the model id in the URL", async () => {
    const { ctx, calls } = fakeCtx(() => json(200, {}), { location: "us-central1" })
    await handle(request({ ...HAIKU, id: "claude-opus-4-5@20251101" }, { max_tokens: 5, messages: [] }), ctx)
    expect(calls[0].url).toBe(
      "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/anthropic/models/claude-opus-4-5@20251101:rawPredict",
    )
  })

  test("does not retry the terms-of-use 400 in another region and converts it to the Anthropic error shape", async () => {
    const message = "Request rejected in accordance with Anthropic's terms of use and policies [Entity type: Billing Account]"
    const { ctx, calls } = fakeCtx(() => json(400, { error: { code: 400, message, status: "FAILED_PRECONDITION" } }))
    const res = await handle(request(HAIKU, { max_tokens: 5, messages: [] }), ctx)
    expect(calls).toHaveLength(1)
    expect(res.status).toBe(400)
    const body = await res.json()
    expect(body.type).toBe("error")
    expect(body.error.message).toBe(message)
  })

  test("unwraps the array-wrapped error streamRawPredict returns", async () => {
    const message = "Request rejected in accordance with Anthropic's terms of use and policies [Entity type: Billing Account]"
    const { ctx } = fakeCtx(() => json(400, [{ error: { code: 400, message, status: "FAILED_PRECONDITION" } }]))
    const res = await handle(request(HAIKU, { max_tokens: 5, stream: true, messages: [] }), ctx)
    expect(res.status).toBe(400)
    const body = await res.json()
    expect(body.type).toBe("error")
    expect(body.error.message).toBe(message)
  })

  test("passes an Anthropic-shaped upstream error through unchanged", async () => {
    const text = '{"type":"error","error":{"type":"invalid_request_error","message":"max_tokens: must be positive"}}'
    const { ctx } = fakeCtx(() => new Response(text, { status: 400, headers: { "content-type": "application/json" } }))
    const res = await handle(request(HAIKU, { max_tokens: 0, messages: [] }), ctx)
    expect(res.status).toBe(400)
    expect(await res.text()).toBe(text)
  })

  test("fails over to the next region on 429", async () => {
    const { ctx, calls } = fakeCtx((i) => (i === 0 ? json(429, { error: { code: 429, message: "quota", status: "RESOURCE_EXHAUSTED" } }) : json(200, { ok: true })))
    const res = await handle(request(HAIKU, { max_tokens: 5, messages: [] }), ctx)
    expect(res.status).toBe(200)
    expect(calls.map((c) => new URL(c.url).host)).toEqual(["aiplatform.googleapis.com", "us-central1-aiplatform.googleapis.com"])
  })
})
