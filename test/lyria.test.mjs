import {afterEach, beforeEach, describe, expect, test} from "bun:test"
import {mkdtemp, readdir, readFile, rm} from "node:fs/promises"
import {tmpdir} from "node:os"
import path from "node:path"
import {pathToFileURL} from "node:url"
import {handle} from "../lib/handlers/lyria.mjs"

let dir

beforeEach(async () => {
  dir = await mkdtemp(path.join(tmpdir(), "lyria-test-"))
})

afterEach(async () => {
  await rm(dir, {recursive: true, force: true})
})

const LYRIA = {id: "lyria-002", family: "lyria", regions: ["us-central1"]}
const WAV = Buffer.from("RIFF-fake-wav-48k").toString("base64")

function fakeFetch(respond) {
  const calls = []
  const fetch = async (url, init = {}) => {
    calls.push({url: String(url), init, body: init.body ? JSON.parse(init.body) : undefined})
    return respond(String(url), init)
  }
  return {fetch, calls}
}

function ctxFor(fetch, options = {}) {
  return {project: "my-project", location: "auto", token: async () => "tok", fetch, mediaDir: dir, options, log: () => {}}
}

function req(text) {
  return {
    protocol: "gemini",
    suffix: "/models/lyria-002:streamGenerateContent?alt=sse",
    entry: LYRIA,
    pinned: undefined,
    apiId: LYRIA.id,
    body: {contents: [{role: "user", parts: [{text}]}]},
    raw: undefined,
    headers: new Headers(),
    stream: true,
    signal: undefined,
  }
}

async function events(res) {
  const raw = await res.text()
  return raw
    .split("\n\n")
    .filter((b) => b.startsWith("data: "))
    .map((b) => JSON.parse(b.slice("data: ".length)))
}

const textOf = (chunk) => chunk.candidates[0].content.parts.map((p) => p.text ?? "").join("")

describe("lyria-002", () => {
  test("posts a single-sample predict body to the us-central1 publisher model", async () => {
    const {fetch, calls} = fakeFetch(() => Response.json({predictions: [{audioContent: WAV, mimeType: "audio/wav"}]}))
    await (await handle(req("calm acoustic folk"), ctxFor(fetch))).text()
    expect(calls).toHaveLength(1)
    expect(calls[0].url).toBe(
      "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/lyria-002:predict",
    )
    expect(calls[0].init.headers.authorization).toBe("Bearer tok")
    expect(calls[0].body).toEqual({instances: [{prompt: "calm acoustic folk"}], parameters: {sample_count: 1}})
  })

  test("forwards ctx.options.lyria.negativePrompt as negative_prompt", async () => {
    const {fetch, calls} = fakeFetch(() => Response.json({predictions: [{audioContent: WAV, mimeType: "audio/wav"}]}))
    await (await handle(req("folk"), ctxFor(fetch, {lyria: {negativePrompt: "drums"}}))).text()
    expect(calls[0].body.instances).toEqual([{prompt: "folk", negative_prompt: "drums"}])
  })

  test("saves the returned WAV byte-for-byte and replies with a file link, path and inline audio", async () => {
    const {fetch} = fakeFetch(() => Response.json({predictions: [{audioContent: WAV, mimeType: "audio/wav"}]}))
    const out = await events(await handle(req("folk"), ctxFor(fetch)))
    const [name] = await readdir(dir)
    expect(name).toMatch(/^lyria-\d{8}-\d{6}-[0-9a-f]{4}\.wav$/)
    const file = path.join(dir, name)
    expect(await readFile(file, "utf8")).toBe("RIFF-fake-wav-48k")
    const final = out.at(-1)
    expect(final.candidates[0].finishReason).toBe("STOP")
    expect(final.usageMetadata).toEqual({promptTokenCount: 0, candidatesTokenCount: 0, totalTokenCount: 0})
    expect(textOf(final)).toContain(`[audio](${pathToFileURL(file).href})`)
    expect(textOf(final)).toContain(`Path: ${file}`)
    expect(final.candidates[0].content.parts[1].inlineData).toEqual({mimeType: "audio/wav", data: WAV})
  })

  test("accepts the bytesBase64Encoded spelling as a fallback", async () => {
    const {fetch} = fakeFetch(() => Response.json({predictions: [{bytesBase64Encoded: WAV}]}))
    await events(await handle(req("folk"), ctxFor(fetch)))
    const [name] = await readdir(dir)
    expect(name).toMatch(/\.wav$/)
    expect(await readFile(path.join(dir, name), "utf8")).toBe("RIFF-fake-wav-48k")
  })

  test("a 2xx reply with no audio is a 502 and nothing is saved", async () => {
    const {fetch} = fakeFetch(() => Response.json({predictions: []}))
    const res = await handle(req("folk"), ctxFor(fetch))
    expect(res.status).toBe(502)
    expect((await res.json()).error.message).toBe("Lyria returned no audio")
    expect(await readdir(dir)).toEqual([])
  })

  test("an upstream rejection returns its status and Google message as an HTTP error", async () => {
    const {fetch} = fakeFetch(() =>
      Response.json({error: {code: 429, message: "Quota exceeded", status: "RESOURCE_EXHAUSTED"}}, {status: 429}),
    )
    const res = await handle(req("folk"), ctxFor(fetch))
    expect(res.status).toBe(429)
    expect((await res.json()).error.message).toBe("Quota exceeded")
    expect(await readdir(dir)).toEqual([])
  })

  test("an unreachable upstream is a 502", async () => {
    const {fetch} = fakeFetch(() => {
      throw new TypeError("socket hang up")
    })
    const res = await handle(req("folk"), ctxFor(fetch))
    expect(res.status).toBe(502)
    expect((await res.json()).error.message).toContain("socket hang up")
  })

  test("blank text is a 400 gemini error and nothing is sent upstream", async () => {
    const {fetch, calls} = fakeFetch(() => Response.json({predictions: []}))
    const res = await handle(req(" \n "), ctxFor(fetch))
    expect(res.status).toBe(400)
    expect(calls).toHaveLength(0)
  })

  test("strips markdown links, urls, emphasis and whitespace runs from the prompt", async () => {
    const {fetch, calls} = fakeFetch(() => Response.json({predictions: [{audioContent: WAV, mimeType: "audio/wav"}]}))
    await handle(req("**Calm** *lo-fi* piano loop\nsee [notes](https://example.com/a.txt) and [audio](file:///tmp/a.wav)"), ctxFor(fetch))
    expect(calls[0].body.instances).toEqual([
      {prompt: "Calm lo-fi piano loop see notes and audio"},
    ])
  })

  test("text that only holds links and whitespace is a 400 and nothing is sent upstream", async () => {
    const {fetch, calls} = fakeFetch(() => Response.json({predictions: []}))
    const res = await handle(req("https://example.com/song.wav\n\n[](file:///tmp/x.wav)"), ctxFor(fetch))
    expect(res.status).toBe(400)
    expect(calls).toHaveLength(0)
  })

  test("a language-rejection 400 keeps Google's message and adds the phrasing hint", async () => {
    const message =
      "Audio generation failed with the following error: Unsupported language detected. Please use one of the supported languages: en. If you intended to use a supported language, please try again with a different prompt."
    const {fetch} = fakeFetch(() => Response.json({error: {code: 400, message, status: "INVALID_ARGUMENT"}}, {status: 400}))
    const res = await handle(req("calm lo-fi piano loop"), ctxFor(fetch))
    expect(res.status).toBe(400)
    const error = (await res.json()).error.message
    expect(error).toContain(message)
    expect(error).toContain("full English sentence")
  })

  test("other 400s keep Google's message verbatim", async () => {
    const {fetch} = fakeFetch(() => Response.json({error: {code: 400, message: "prompt is too long", status: "INVALID_ARGUMENT"}}, {status: 400}))
    const res = await handle(req("folk"), ctxFor(fetch))
    expect(res.status).toBe(400)
    expect((await res.json()).error.message).toBe("prompt is too long")
  })
})
