import {afterEach, beforeEach, describe, expect, test} from "bun:test"
import {mkdtemp, readdir, readFile, rm} from "node:fs/promises"
import {tmpdir} from "node:os"
import path from "node:path"
import {fileURLToPath, pathToFileURL} from "node:url"
import {extFor, parsePcmMime, pcmToWav, saveMedia} from "../lib/media.mjs"
import {lastUserTurn} from "../lib/prompt.mjs"
import {chunk, geminiStream} from "../lib/sse.mjs"

let tmp

beforeEach(async () => {
  tmp = await mkdtemp(path.join(tmpdir(), "media-test-"))
})

afterEach(async () => {
  await rm(tmp, {recursive: true, force: true})
})

const ascii = (bytes, from, to) => new TextDecoder().decode(bytes.subarray(from, to))

async function events(res) {
  const text = await res.text()
  return text
    .split("\n\n")
    .filter(Boolean)
    .map((block) => (block === ": keepalive" ? {keepalive: true} : JSON.parse(block.replace(/^data: /, ""))))
}

describe("pcmToWav", () => {
  test("writes a canonical 44-byte mono 16-bit header in front of the samples", () => {
    const pcm = new Uint8Array([1, 2, 3, 4])
    const wav = pcmToWav(pcm, {rate: 24000, channels: 1, bits: 16})
    const view = new DataView(wav.buffer)
    expect(wav.byteLength).toBe(48)
    expect(ascii(wav, 0, 4)).toBe("RIFF")
    expect(view.getUint32(4, true)).toBe(36 + 4)
    expect(ascii(wav, 8, 12)).toBe("WAVE")
    expect(ascii(wav, 12, 16)).toBe("fmt ")
    expect(view.getUint32(16, true)).toBe(16)
    expect(view.getUint16(20, true)).toBe(1)
    expect(view.getUint16(22, true)).toBe(1)
    expect(view.getUint32(24, true)).toBe(24000)
    expect(view.getUint32(28, true)).toBe(48000)
    expect(view.getUint16(32, true)).toBe(2)
    expect(view.getUint16(34, true)).toBe(16)
    expect(ascii(wav, 36, 40)).toBe("data")
    expect(view.getUint32(40, true)).toBe(4)
    expect(Array.from(wav.subarray(44))).toEqual([1, 2, 3, 4])
  })

  test("derives block align and byte rate from channels and bits", () => {
    const wav = pcmToWav(new Uint8Array(8), {rate: 16000, channels: 2, bits: 16})
    const view = new DataView(wav.buffer)
    expect(view.getUint16(22, true)).toBe(2)
    expect(view.getUint32(28, true)).toBe(16000 * 4)
    expect(view.getUint16(32, true)).toBe(4)
    expect(view.getUint32(40, true)).toBe(8)
  })

  test("defaults to 24 kHz mono 16-bit", () => {
    const view = new DataView(pcmToWav(new Uint8Array(2)).buffer)
    expect(view.getUint32(24, true)).toBe(24000)
    expect(view.getUint16(22, true)).toBe(1)
    expect(view.getUint16(34, true)).toBe(16)
  })
})

describe("parsePcmMime", () => {
  test("reads rate and channels from Gemini TTS mime types", () => {
    expect(parsePcmMime("audio/L16;codec=pcm;rate=24000")).toEqual({rate: 24000, channels: 1})
    expect(parsePcmMime("audio/l16; rate=24000; channels=2")).toEqual({rate: 24000, channels: 2})
  })

  test("returns null for non-PCM audio and bad rates", () => {
    expect(parsePcmMime("audio/wav")).toBeNull()
    expect(parsePcmMime("audio/L16;codec=opus;rate=24000")).toBeNull()
    expect(parsePcmMime("audio/L16;rate=abc")).toBeNull()
  })
})

describe("extFor", () => {
  test("maps known mimes, ignores parameters, and falls back to bin", () => {
    expect(extFor("image/png")).toBe("png")
    expect(extFor("audio/wav; charset=x")).toBe("wav")
    expect(extFor("audio/mpeg")).toBe("mp3")
    expect(extFor("video/mp4")).toBe("mp4")
    expect(extFor("application/x-unknown")).toBe("bin")
    expect(extFor("constructor")).toBe("bin")
  })
})

describe("saveMedia", () => {
  test("writes decoded base64 under prefix-timestamp-rand.ext and returns path and file URL", async () => {
    const dir = path.join(tmp, "media dir")
    const saved = await saveMedia({mediaDir: dir}, {data: Buffer.from("RIFFxxxx").toString("base64"), mime: "audio/wav", prefix: "tts"})
    expect(path.dirname(saved.path)).toBe(dir)
    expect(path.basename(saved.path)).toMatch(/^tts-\d{8}-\d{6}-[0-9a-f]{4}\.wav$/)
    expect(path.isAbsolute(saved.path)).toBe(true)
    expect(saved.url).toBe(pathToFileURL(saved.path).href)
    expect(saved.url.startsWith("file:///")).toBe(true)
    expect(saved.url).toContain("media%20dir")
    expect(fileURLToPath(saved.url)).toBe(saved.path)
    expect(saved.bytes).toBe(8)
    expect(ascii(new Uint8Array(await readFile(saved.path)), 0, 8)).toBe("RIFFxxxx")
  })

  test("accepts raw bytes and defaults the prefix to vertex", async () => {
    const saved = await saveMedia({mediaDir: tmp}, {data: new Uint8Array([9, 8, 7]), mime: "video/mp4"})
    expect(path.basename(saved.path)).toMatch(/^vertex-\d{8}-\d{6}-[0-9a-f]{4}\.mp4$/)
    expect(Array.from(new Uint8Array(await readFile(saved.path)))).toEqual([9, 8, 7])
    expect(saved.bytes).toBe(3)
  })

  test("creates the media directory on demand", async () => {
    const dir = path.join(tmp, "a", "b")
    await saveMedia({mediaDir: dir}, {data: "AA==", mime: "image/png"})
    expect((await readdir(dir)).length).toBe(1)
  })
})

describe("lastUserTurn", () => {
  test("takes the newest user turn, joins its text parts, and buckets media by type sorted by mime", () => {
    const body = {
      contents: [
        {role: "user", parts: [{text: "old"}]},
        {role: "model", parts: [{text: "reply"}]},
        {
          role: "user",
          parts: [
            {text: "first line"},
            {inlineData: {mimeType: "image/png", data: "P"}},
            {inlineData: {mimeType: "image/jpeg", data: "J"}},
            {inlineData: {mimeType: "audio/wav", data: "A"}},
            {inlineData: {mimeType: "video/mp4", data: "V"}},
            {inlineData: {mimeType: "application/pdf", data: "D"}},
            {text: "second line"},
          ],
        },
      ],
    }
    expect(lastUserTurn(body)).toEqual({
      text: "first line\nsecond line",
      images: [
        {mime: "image/jpeg", data: "J"},
        {mime: "image/png", data: "P"},
      ],
      audio: [{mime: "audio/wav", data: "A"}],
      video: [{mime: "video/mp4", data: "V"}],
    })
  })

  test("returns empty fields for a missing body or no user turn", () => {
    const empty = {text: "", images: [], audio: [], video: []}
    expect(lastUserTurn(null)).toEqual(empty)
    expect(lastUserTurn({contents: [{role: "model", parts: [{text: "hi"}]}]})).toEqual(empty)
  })
})

describe("geminiStream", () => {
  test("serves text/event-stream with data lines, keepalive comments and an OTHER finish for thrown errors", async () => {
    const res = geminiStream(async ({emit, keepalive}) => {
      keepalive()
      emit(chunk({text: "working", thought: true}))
      throw new Error("boom")
    })
    expect(res.headers.get("content-type")).toBe("text/event-stream")
    const raw = await res.text()
    expect(raw.startsWith(": keepalive\n\ndata: ")).toBe(true)
    const [progress, failure] = raw.split("\n\n").filter((b) => b.startsWith("data: ")).map((b) => JSON.parse(b.slice("data: ".length)))
    expect(progress.candidates[0].content.parts).toEqual([{text: "working", thought: true}])
    expect(failure.candidates[0].finishReason).toBe("OTHER")
    expect(failure.candidates[0].content.parts).toEqual([{text: "Error: boom"}])
  })

  test("final chunks carry finishReason, usage and inlineData when given", async () => {
    const res = geminiStream(async ({emit}) => {
      emit(chunk({text: "done", inlineData: {mimeType: "audio/wav", data: "AA=="}, finishReason: "STOP", usage: {totalTokenCount: 0}}))
    })
    const [only] = await events(res)
    expect(only.candidates[0]).toEqual({
      index: 0,
      content: {role: "model", parts: [{text: "done"}, {inlineData: {mimeType: "audio/wav", data: "AA=="}}]},
      finishReason: "STOP",
    })
    expect(only.usageMetadata).toEqual({totalTokenCount: 0})
  })

  test("stops writing and closes when the request signal aborts", async () => {
    const ac = new AbortController()
    const res = geminiStream(
      async ({emit, signal}) => {
        emit(chunk({text: "first"}))
        await new Promise((_, reject) => signal.addEventListener("abort", () => reject(new Error("aborted")), {once: true}))
      },
      {signal: ac.signal},
    )
    const reader = res.body.getReader()
    const first = await reader.read()
    expect(first.done).toBe(false)
    ac.abort()
    expect((await reader.read()).done).toBe(true)
  })
})
