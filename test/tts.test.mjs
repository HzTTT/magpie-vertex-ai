import {afterEach, beforeEach, describe, expect, test} from "bun:test"
import {mkdtemp, readdir, readFile, rm} from "node:fs/promises"
import {tmpdir} from "node:os"
import path from "node:path"
import {pathToFileURL} from "node:url"
import {handle} from "../lib/handlers/tts.mjs"

let dir

beforeEach(async () => {
  dir = await mkdtemp(path.join(tmpdir(), "tts-test-"))
})

afterEach(async () => {
  await rm(dir, {recursive: true, force: true})
})

const GEMINI_TTS = {id: "gemini-2.5-flash-tts", family: "gemini-tts", regions: ["us-central1", "europe-west4"]}
const CHIRP = {id: "chirp3-hd", family: "chirp-tts", regions: ["global"]}

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

function req(entry, text) {
  return {
    protocol: "gemini",
    suffix: "/models/x:streamGenerateContent?alt=sse",
    entry,
    pinned: undefined,
    apiId: entry.id,
    body: {contents: [{role: "user", parts: [{text}]}]},
    raw: undefined,
    headers: new Headers(),
    stream: true,
    signal: undefined,
  }
}

async function events(res) {
  const text = await res.text()
  return text
    .split("\n\n")
    .filter((b) => b.startsWith("data: "))
    .map((b) => JSON.parse(b.slice("data: ".length)))
}

const textOf = (chunk) => chunk.candidates[0].content.parts.map((p) => p.text ?? "").join("")

const PCM_B64 = Buffer.from([1, 0, 2, 0]).toString("base64")
const geminiAudio = () =>
  Response.json({
    candidates: [{content: {role: "model", parts: [{inlineData: {mimeType: "audio/L16;codec=pcm;rate=24000", data: PCM_B64}}]}}],
  })

describe("gemini-tts", () => {
  test("posts a non-stream AUDIO generateContent with the default voice", async () => {
    const {fetch, calls} = fakeFetch(geminiAudio)
    await (await handle(req(GEMINI_TTS, "Hello there"), ctxFor(fetch))).text()
    expect(calls).toHaveLength(1)
    expect(calls[0].url).toBe(
      "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-2.5-flash-tts:generateContent",
    )
    expect(calls[0].init.headers.authorization).toBe("Bearer tok")
    expect(calls[0].body).toEqual({
      contents: [{role: "user", parts: [{text: "Hello there"}]}],
      generationConfig: {
        responseModalities: ["AUDIO"],
        speechConfig: {voiceConfig: {prebuiltVoiceConfig: {voiceName: "Kore"}}},
      },
    })
  })

  test("uses ctx.options.ttsVoice when set", async () => {
    const {fetch, calls} = fakeFetch(geminiAudio)
    await (await handle(req(GEMINI_TTS, "hi"), ctxFor(fetch, {ttsVoice: "Puck"}))).text()
    expect(calls[0].body.generationConfig.speechConfig.voiceConfig.prebuiltVoiceConfig.voiceName).toBe("Puck")
  })

  test("saves the PCM as a WAV file and replies with a file link, path and inline audio", async () => {
    const {fetch} = fakeFetch(geminiAudio)
    const out = await events(await handle(req(GEMINI_TTS, "Hello there"), ctxFor(fetch)))
    const [wavName] = await readdir(dir)
    expect(wavName).toMatch(/^tts-\d{8}-\d{6}-[0-9a-f]{4}\.wav$/)
    const wavPath = path.join(dir, wavName)
    const wav = new Uint8Array(await readFile(wavPath))
    expect(new TextDecoder().decode(wav.subarray(0, 4))).toBe("RIFF")
    expect(new DataView(wav.buffer).getUint32(24, true)).toBe(24000)
    expect(Array.from(wav.subarray(44))).toEqual([1, 0, 2, 0])

    const final = out.at(-1)
    expect(final.candidates[0].finishReason).toBe("STOP")
    expect(final.usageMetadata).toEqual({promptTokenCount: 0, candidatesTokenCount: 0, totalTokenCount: 0})
    expect(textOf(final)).toContain(`[audio](${pathToFileURL(wavPath).href})`)
    expect(textOf(final)).toContain(`Path: ${wavPath}`)
    expect(final.candidates[0].content.parts[1].inlineData).toEqual({mimeType: "audio/wav", data: Buffer.from(wav).toString("base64")})
  })

  test("fails over to the next region on a retryable status", async () => {
    const {fetch, calls} = fakeFetch((url) => (url.includes("us-central1") ? new Response("RESOURCE_EXHAUSTED", {status: 429}) : geminiAudio()))
    await (await handle(req(GEMINI_TTS, "hi"), ctxFor(fetch))).text()
    expect(calls.map((c) => c.url.split("/")[2])).toEqual(["us-central1-aiplatform.googleapis.com", "europe-west4-aiplatform.googleapis.com"])
  })

  test("a rejected synthesis returns the upstream status and message as an HTTP error, saving nothing", async () => {
    const {fetch} = fakeFetch(() => new Response("bad voice", {status: 400}))
    const res = await handle(req(GEMINI_TTS, "hi"), ctxFor(fetch))
    expect(res.status).toBe(400)
    expect((await res.json()).error.message).toBe("bad voice")
    expect(await readdir(dir)).toEqual([])
  })

  test("a Google error body keeps its status and message", async () => {
    const {fetch} = fakeFetch(() =>
      Response.json({error: {code: 403, message: "Permission denied", status: "PERMISSION_DENIED"}}, {status: 403}),
    )
    const res = await handle(req(GEMINI_TTS, "hi"), ctxFor(fetch))
    expect(res.status).toBe(403)
    expect((await res.json()).error.message).toBe("Permission denied")
  })

  test("a 2xx reply without audio is a 502", async () => {
    const {fetch} = fakeFetch(() => Response.json({candidates: [{content: {role: "model", parts: [{text: "hi"}]}}]}))
    const res = await handle(req(GEMINI_TTS, "hi"), ctxFor(fetch))
    expect(res.status).toBe(502)
    expect((await res.json()).error.message).toBe("Gemini TTS returned no audio")
  })

  test("an unexpected audio type is a 502", async () => {
    const {fetch} = fakeFetch(() =>
      Response.json({candidates: [{content: {role: "model", parts: [{inlineData: {mimeType: "audio/mpeg", data: PCM_B64}}]}}]}),
    )
    const res = await handle(req(GEMINI_TTS, "hi"), ctxFor(fetch))
    expect(res.status).toBe(502)
    expect((await res.json()).error.message).toBe("Unexpected Gemini TTS audio type: audio/mpeg")
  })

  test("an unreachable upstream is a 502", async () => {
    const {fetch} = fakeFetch(() => {
      throw new TypeError("socket hang up")
    })
    const res = await handle(req(GEMINI_TTS, "hi"), ctxFor(fetch))
    expect(res.status).toBe(502)
    expect((await res.json()).error.message).toContain("socket hang up")
  })
})

describe("chirp-tts", () => {
  const mp3 = () => Response.json({audioContent: Buffer.from("ID3-mp3-bytes").toString("base64")})

  test("synthesizes with the Cloud TTS endpoint, user-project header and an English Chirp voice", async () => {
    const {fetch, calls} = fakeFetch(mp3)
    await (await handle(req(CHIRP, "Hello world"), ctxFor(fetch))).text()
    expect(calls).toHaveLength(1)
    expect(calls[0].url).toBe("https://texttospeech.googleapis.com/v1/text:synthesize")
    expect(calls[0].init.headers["x-goog-user-project"]).toBe("my-project")
    expect(calls[0].init.headers.authorization).toBe("Bearer tok")
    expect(calls[0].body).toEqual({
      input: {text: "Hello world"},
      voice: {languageCode: "en-US", name: "en-US-Chirp3-HD-Achernar"},
      audioConfig: {audioEncoding: "MP3"},
    })
  })

  test("picks cmn-CN when the text contains CJK characters", async () => {
    const {fetch, calls} = fakeFetch(mp3)
    await (await handle(req(CHIRP, "你好，世界"), ctxFor(fetch))).text()
    expect(calls[0].body.voice).toEqual({languageCode: "cmn-CN", name: "cmn-CN-Chirp3-HD-Achernar"})
  })

  test("honours chirpLanguage and chirpVoice options", async () => {
    const {fetch, calls} = fakeFetch(mp3)
    await (await handle(req(CHIRP, "你好"), ctxFor(fetch, {chirpLanguage: "en-GB", chirpVoice: "Puck"}))).text()
    expect(calls[0].body.voice).toEqual({languageCode: "en-GB", name: "en-GB-Chirp3-HD-Puck"})
  })

  test("saves MP3 and replies with the link, path and audio/mpeg inline data", async () => {
    const {fetch} = fakeFetch(mp3)
    const out = await events(await handle(req(CHIRP, "Hello"), ctxFor(fetch)))
    const [name] = await readdir(dir)
    expect(name).toMatch(/^tts-\d{8}-\d{6}-[0-9a-f]{4}\.mp3$/)
    expect(await readFile(path.join(dir, name), "utf8")).toBe("ID3-mp3-bytes")
    const final = out.at(-1)
    expect(textOf(final)).toContain(`[audio](${pathToFileURL(path.join(dir, name)).href})`)
    expect(final.candidates[0].content.parts[1].inlineData.mimeType).toBe("audio/mpeg")
  })

  test("a rejected synthesis returns the upstream status and Google message, saving nothing", async () => {
    const {fetch} = fakeFetch(() =>
      Response.json({error: {code: 403, message: "Cloud Text-to-Speech API has not been used", status: "PERMISSION_DENIED"}}, {status: 403}),
    )
    const res = await handle(req(CHIRP, "Hello"), ctxFor(fetch))
    expect(res.status).toBe(403)
    expect((await res.json()).error.message).toBe("Cloud Text-to-Speech API has not been used")
    expect(await readdir(dir)).toEqual([])
  })

  test("a 2xx reply without audio is a 502", async () => {
    const {fetch} = fakeFetch(() => Response.json({}))
    const res = await handle(req(CHIRP, "Hello"), ctxFor(fetch))
    expect(res.status).toBe(502)
    expect((await res.json()).error.message).toBe("Chirp TTS returned no audio")
  })

  test("an unreachable upstream is a 502", async () => {
    const {fetch} = fakeFetch(() => {
      throw new TypeError("socket hang up")
    })
    const res = await handle(req(CHIRP, "Hello"), ctxFor(fetch))
    expect(res.status).toBe(502)
    expect((await res.json()).error.message).toContain("socket hang up")
  })
})

describe("input validation", () => {
  test("blank text is a 400 gemini error and nothing is sent upstream", async () => {
    const {fetch, calls} = fakeFetch(geminiAudio)
    const res = await handle(req(GEMINI_TTS, "   "), ctxFor(fetch))
    expect(res.status).toBe(400)
    expect(calls).toHaveLength(0)
  })
})
