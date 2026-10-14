import { describe, expect, test } from "bun:test"
import { handle } from "../lib/handlers/transcribe.mjs"

const json = (obj, status = 200) =>
  new Response(JSON.stringify(obj), { status, headers: { "content-type": "application/json" } })

const reply = text => json({ candidates: [{ content: { role: "model", parts: [{ text }] } }] })

const SEGMENTS = JSON.stringify({
  language: "en",
  segments: [
    { start: 0, end: 1.5, text: "Hello" },
    { start: 1.5, end: 3.25, text: "world" },
  ],
})

function harness(respond) {
  const calls = []
  const ctx = {
    project: "my-project",
    location: "auto",
    token: async () => "test-token",
    fetch: async (url, init) => {
      const body = JSON.parse(init.body)
      calls.push({ url, auth: init.headers.authorization, body })
      return respond(url, body, calls.length)
    },
    mediaDir: "/tmp/unused",
    options: {},
    log: () => {},
  }
  return { ctx, calls }
}

const gemini = { id: "gemini-2.5-flash", key: "gemini-2.5-flash", publisher: "google", family: "gemini", regions: ["global"] }

function request(form, suffix = "/audio/transcriptions", entry = gemini) {
  return {
    protocol: "openai",
    suffix,
    entry,
    pinned: undefined,
    apiId: entry.id,
    body: null,
    raw: null,
    headers: new Headers(),
    stream: false,
    signal: undefined,
    form,
  }
}

function form(fields) {
  const f = new FormData()
  for (const [k, v] of Object.entries(fields)) f.append(k, v)
  return f
}

const wav = (bytes = [1, 2, 3, 4]) => new File([new Uint8Array(bytes)], "clip.wav", { type: "audio/wav" })

describe("plain transcription (json / text)", () => {
  test("sends the file as inlineData with language and prompt context and returns {text}", async () => {
    const { ctx, calls } = harness(() => reply("hello world"))
    const res = await handle(request(form({ file: wav(), model: "gemini-2.5-flash", language: "en", prompt: "Names: Ada" })), ctx)

    expect(res.status).toBe(200)
    expect(calls[0].url).toBe(
      "https://aiplatform.googleapis.com/v1/projects/my-project/locations/global/publishers/google/models/gemini-2.5-flash:generateContent",
    )
    expect(calls[0].auth).toBe("Bearer test-token")
    const { role, parts } = calls[0].body.contents[0]
    expect(role).toBe("user")
    expect(parts[0].inlineData).toEqual({ mimeType: "audio/wav", data: Buffer.from([1, 2, 3, 4]).toString("base64") })
    expect(parts[1].text).toContain("language: en.")
    expect(parts[1].text).toContain("Names: Ada")
    expect(parts[1].text).toContain("Output only the transcript.")
    expect(calls[0].body.generationConfig.responseSchema).toBeUndefined()
    expect(await res.json()).toEqual({ text: "hello world" })
  })

  test("response_format text returns the transcript as text/plain", async () => {
    const { ctx } = harness(() => reply("hello world"))
    const res = await handle(request(form({ file: wav(), response_format: "text" })), ctx)
    expect(res.headers.get("content-type")).toMatch(/^text\/plain/)
    expect(await res.text()).toBe("hello world")
  })

  test("forwards temperature into generationConfig", async () => {
    const { ctx, calls } = harness(() => reply("ok"))
    await handle(request(form({ file: wav(), temperature: "0.2" })), ctx)
    expect(calls[0].body.generationConfig.temperature).toBe(0.2)
  })

  test("guesses the audio type from the file name when the part has no type", async () => {
    const { ctx, calls } = harness(() => reply("ok"))
    await handle(request(form({ file: new File([new Uint8Array([9])], "talk.mp3") })), ctx)
    expect(calls[0].body.contents[0].parts[0].inlineData.mimeType).toBe("audio/mp3")
  })
})

describe("segment formats (srt / vtt / verbose_json)", () => {
  test("srt asks Gemini for JSON segments and renders SRT", async () => {
    const { ctx, calls } = harness(() => reply(SEGMENTS))
    const res = await handle(request(form({ file: wav(), response_format: "srt" })), ctx)

    expect(calls[0].body.generationConfig.responseMimeType).toBe("application/json")
    expect(calls[0].body.generationConfig.responseSchema.required).toEqual(["segments"])
    expect(res.headers.get("content-type")).toMatch(/^text\/plain/)
    expect(await res.text()).toBe(
      "1\n00:00:00,000 --> 00:00:01,500\nHello\n\n2\n00:00:01,500 --> 00:00:03,250\nworld\n",
    )
  })

  test("vtt renders WebVTT with dot-separated milliseconds", async () => {
    const { ctx } = harness(() => reply(SEGMENTS))
    const res = await handle(request(form({ file: wav(), response_format: "vtt" })), ctx)
    expect(res.headers.get("content-type")).toMatch(/^text\/vtt/)
    expect(await res.text()).toBe(
      "WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nHello\n\n00:00:01.500 --> 00:00:03.250\nworld\n",
    )
  })

  test("verbose_json returns task, language, duration, text and numbered segments", async () => {
    const { ctx } = harness(() => reply(SEGMENTS))
    const res = await handle(request(form({ file: wav(), response_format: "verbose_json" })), ctx)
    expect(await res.json()).toEqual({
      task: "transcribe",
      language: "en",
      duration: 3.25,
      text: "Hello world",
      segments: [
        { id: 0, start: 0, end: 1.5, text: "Hello" },
        { id: 1, start: 1.5, end: 3.25, text: "world" },
      ],
    })
  })

  test("reports a 502 when the segment JSON is malformed", async () => {
    const { ctx } = harness(() => reply("not json"))
    const res = await handle(request(form({ file: wav(), response_format: "srt" })), ctx)
    expect(res.status).toBe(502)
  })
})

describe("translations endpoint", () => {
  test("/audio/translations asks for English output and reports task translate", async () => {
    const { ctx, calls } = harness(() => reply(SEGMENTS))
    const res = await handle(request(form({ file: wav(), response_format: "verbose_json" }), "/audio/translations"), ctx)

    expect(calls[0].body.contents[0].parts[1].text).toContain("Translate the speech in this audio into English.")
    const body = await res.json()
    expect(body.task).toBe("translate")
    expect(body.language).toBe("english")
  })
})

describe("input checks and failover", () => {
  test("rejects files over 15 MB with 413 before calling Vertex", async () => {
    const { ctx, calls } = harness(() => reply("x"))
    const big = new File([new Uint8Array(15 * 1024 * 1024 + 1)], "big.wav", { type: "audio/wav" })
    const res = await handle(request(form({ file: big })), ctx)
    expect(res.status).toBe(413)
    expect(calls).toHaveLength(0)
  })

  test("requires a file field", async () => {
    const { ctx, calls } = harness(() => reply("x"))
    const res = await handle(request(form({ model: "gemini-2.5-flash" })), ctx)
    expect(res.status).toBe(400)
    expect(calls).toHaveLength(0)
  })

  test("rejects an unknown response_format", async () => {
    const { ctx, calls } = harness(() => reply("x"))
    const res = await handle(request(form({ file: wav(), response_format: "mp3" })), ctx)
    expect(res.status).toBe(400)
    expect(calls).toHaveLength(0)
  })

  test("fails over to the next region on a retryable error", async () => {
    const { ctx, calls } = harness((url) =>
      url.includes("/locations/global/")
        ? json({ error: { code: 429, message: "RESOURCE_EXHAUSTED" } }, 429)
        : reply("hi"),
    )
    const res = await handle(request(form({ file: wav() }), "/audio/transcriptions", { ...gemini, regions: ["global", "us-central1"] }), ctx)

    expect(calls.map(c => c.url.split("/locations/")[1].split("/")[0])).toEqual(["global", "us-central1"])
    expect(await res.json()).toEqual({ text: "hi" })
  })
})
