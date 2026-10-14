import {afterEach, beforeEach, describe, expect, test} from "bun:test"
import {mkdtemp, readdir, readFile, rm} from "node:fs/promises"
import {tmpdir} from "node:os"
import path from "node:path"
import {pathToFileURL} from "node:url"
import {handle} from "../lib/handlers/veo.mjs"

let dir

beforeEach(async () => {
  dir = await mkdtemp(path.join(tmpdir(), "veo-test-"))
})

afterEach(async () => {
  await rm(dir, {recursive: true, force: true})
})

const VEO3 = {id: "veo-3.1-generate-001", family: "veo", regions: ["us-central1"]}
const VEO2 = {id: "veo-2.0-generate-001", family: "veo", regions: ["us-central1", "europe-west3"]}
const OP = "projects/my-project/locations/us-central1/publishers/google/models/veo-3.1-generate-001/operations/123"
const MP4 = Buffer.from("fake-mp4-bytes").toString("base64")

// Routes Veo submit and fetchPredictOperation. `polls` are returned in order.
function veoFetch(polls, {submit = () => Response.json({name: OP})} = {}) {
  const calls = []
  const queue = [...polls]
  const fetch = async (url, init = {}) => {
    calls.push({url: String(url), init, body: init.body ? JSON.parse(init.body) : undefined})
    if (String(url).endsWith(":predictLongRunning")) return submit(String(url), init)
    if (String(url).endsWith(":fetchPredictOperation")) return Response.json(queue.shift())
    throw new Error(`unexpected ${url}`)
  }
  return {fetch, calls}
}

function ctxFor(fetch, options = {}) {
  return {
    project: "my-project",
    location: "auto",
    token: async () => "tok",
    fetch,
    mediaDir: dir,
    options: {veoPollMs: 1, ...options},
    log: () => {},
  }
}

// `images` are base64 inlineData parts after the prompt text.
function req(entry, text, images = []) {
  const parts = [{text}, ...images.map((data) => ({inlineData: {mimeType: "image/png", data}}))]
  return {
    protocol: "gemini",
    suffix: "/models/x:streamGenerateContent?alt=sse",
    entry,
    pinned: undefined,
    apiId: entry.id,
    body: {contents: [{role: "user", parts}]},
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
    .filter(Boolean)
    .map((block) => (block === ": keepalive" ? {keepalive: true} : JSON.parse(block.slice("data: ".length))))
}

const pollCalls = (calls) => calls.filter((c) => c.url.endsWith(":fetchPredictOperation"))
const textOf = (chunk) => chunk.candidates[0].content.parts.map((p) => p.text ?? "").join("")
const thoughts = (out) => out.filter((e) => e.candidates?.[0].content.parts[0]?.thought).map(textOf)

describe("submit body", () => {
  test("veo-3 sends first image as image, second as lastFrame, and resolution plus generateAudio", async () => {
    const {fetch, calls} = veoFetch([{done: true, response: {videos: [{bytesBase64Encoded: MP4, mimeType: "video/mp4"}]}}])
    await events(await handle(req(VEO3, "a cat", ["FIRST", "LAST"]), ctxFor(fetch)))
    expect(calls[0].url).toBe(
      "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/veo-3.1-generate-001:predictLongRunning",
    )
    expect(calls[0].init.headers.authorization).toBe("Bearer tok")
    expect(calls[0].body).toEqual({
      instances: [
        {
          prompt: "a cat",
          image: {bytesBase64Encoded: "FIRST", mimeType: "image/png"},
          lastFrame: {bytesBase64Encoded: "LAST", mimeType: "image/png"},
        },
      ],
      parameters: {
        sampleCount: 1,
        durationSeconds: 8,
        aspectRatio: "16:9",
        personGeneration: "allow_adult",
        resolution: "720p",
        generateAudio: true,
      },
    })
  })

  test("frames keep document order even when their mime types differ", async () => {
    const {fetch, calls} = veoFetch([{done: true, response: {videos: [{bytesBase64Encoded: MP4, mimeType: "video/mp4"}]}}])
    const request = req(VEO3, "a cat")
    request.body.contents[0].parts.push(
      {inlineData: {mimeType: "image/png", data: "FIRST"}},
      {inlineData: {mimeType: "image/jpeg", data: "LAST"}},
    )
    await events(await handle(request, ctxFor(fetch)))
    expect(calls[0].body.instances[0].image).toEqual({bytesBase64Encoded: "FIRST", mimeType: "image/png"})
    expect(calls[0].body.instances[0].lastFrame).toEqual({bytesBase64Encoded: "LAST", mimeType: "image/jpeg"})
  })

  test("veo-2 omits resolution, generateAudio and lastFrame", async () => {
    const {fetch, calls} = veoFetch([{done: true, response: {videos: [{bytesBase64Encoded: MP4, mimeType: "video/mp4"}]}}])
    await events(await handle(req(VEO2, "a dog", ["FIRST", "LAST"]), ctxFor(fetch)))
    expect(calls[0].body.instances[0]).toEqual({prompt: "a dog", image: {bytesBase64Encoded: "FIRST", mimeType: "image/png"}})
    expect(calls[0].body.parameters).toEqual({sampleCount: 1, durationSeconds: 8, aspectRatio: "16:9", personGeneration: "allow_adult"})
  })

  test("text-only prompts send no image", async () => {
    const {fetch, calls} = veoFetch([{done: true, response: {videos: [{bytesBase64Encoded: MP4, mimeType: "video/mp4"}]}}])
    await events(await handle(req(VEO3, "sunrise"), ctxFor(fetch)))
    expect(calls[0].body.instances).toEqual([{prompt: "sunrise"}])
  })

  test("ctx.options.veo overrides defaults and negativePrompt is forwarded", async () => {
    const {fetch, calls} = veoFetch([{done: true, response: {videos: [{bytesBase64Encoded: MP4, mimeType: "video/mp4"}]}}])
    const options = {veo: {durationSeconds: 5, aspectRatio: "9:16", resolution: "1080p", generateAudio: false, negativePrompt: "blurry"}}
    await events(await handle(req(VEO3, "x"), ctxFor(fetch, options)))
    expect(calls[0].body.parameters).toEqual({
      sampleCount: 1,
      durationSeconds: 5,
      aspectRatio: "9:16",
      personGeneration: "allow_adult",
      resolution: "1080p",
      generateAudio: false,
      negativePrompt: "blurry",
    })
  })

  test("blank text is a 400 gemini error and nothing is sent upstream", async () => {
    const {fetch, calls} = veoFetch([])
    const res = await handle(req(VEO3, "  "), ctxFor(fetch))
    expect(res.status).toBe(400)
    expect(calls).toHaveLength(0)
  })
})

describe("polling", () => {
  test("polls fetchPredictOperation with the operation name, emitting keepalive and thought progress per poll", async () => {
    const {fetch, calls} = veoFetch([
      {done: false},
      {done: false},
      {done: true, response: {videos: [{bytesBase64Encoded: MP4, mimeType: "video/mp4"}]}},
    ])
    const out = await events(await handle(req(VEO3, "a cat"), ctxFor(fetch)))
    const polls = pollCalls(calls)
    expect(polls).toHaveLength(3)
    expect(polls[0].url).toBe(
      "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/veo-3.1-generate-001:fetchPredictOperation",
    )
    expect(polls.every((p) => p.body.operationName === OP)).toBe(true)
    expect(out.filter((e) => e.keepalive)).toHaveLength(3)
    const progress = thoughts(out)
    expect(progress[0]).toMatch(/^Veo job submitted/)
    expect(progress.filter((t) => t.startsWith("Veo generating"))).toHaveLength(2)
  })

  test("saves the mp4 and replies with a file link and path, finishing with STOP and zero usage", async () => {
    const {fetch} = veoFetch([{done: true, response: {videos: [{bytesBase64Encoded: MP4, mimeType: "video/mp4"}]}}])
    const out = await events(await handle(req(VEO3, "a cat"), ctxFor(fetch)))
    const [name] = await readdir(dir)
    expect(name).toMatch(/^veo-\d{8}-\d{6}-[0-9a-f]{4}\.mp4$/)
    const file = path.join(dir, name)
    expect(await readFile(file, "utf8")).toBe("fake-mp4-bytes")
    const final = out.at(-1)
    expect(final.candidates[0].finishReason).toBe("STOP")
    expect(final.usageMetadata).toEqual({promptTokenCount: 0, candidatesTokenCount: 0, totalTokenCount: 0})
    expect(textOf(final)).toContain(`[video](${pathToFileURL(file).href})`)
    expect(textOf(final)).toContain(`Path: ${file}`)
  })

  test("a gcsUri output is reported in text and nothing is downloaded", async () => {
    const {fetch} = veoFetch([{done: true, response: {videos: [{gcsUri: "gs://bucket/out.mp4", mimeType: "video/mp4"}]}}])
    const out = await events(await handle(req(VEO3, "a cat"), ctxFor(fetch)))
    expect(textOf(out.at(-1))).toContain("gs://bucket/out.mp4")
    expect(await readdir(dir)).toEqual([])
  })

  test("RAI-filtered output replies with the reasons and saves nothing", async () => {
    const {fetch} = veoFetch([
      {done: true, response: {raiMediaFilteredCount: 1, raiMediaFilteredReasons: ["Content blocked by safety filter"]}},
    ])
    const out = await events(await handle(req(VEO3, "a cat"), ctxFor(fetch)))
    const final = out.at(-1)
    expect(final.candidates[0].finishReason).toBe("SAFETY")
    expect(textOf(final)).toContain("1 output(s) removed by Responsible AI filters: Content blocked by safety filter")
    expect(await readdir(dir)).toEqual([])
  })

  test("reads a payload nested under generateVideoResponse", async () => {
    const {fetch} = veoFetch([{done: true, response: {generateVideoResponse: {raiMediaFilteredCount: 1, raiMediaFilteredReasons: ["audio filtered"]}}}])
    const out = await events(await handle(req(VEO3, "a cat"), ctxFor(fetch)))
    expect(out.at(-1).candidates[0].finishReason).toBe("SAFETY")
    expect(textOf(out.at(-1))).toContain("Responsible AI filters: audio filtered")
  })

  test("an operation error is surfaced in the stream as an OTHER error chunk", async () => {
    const {fetch} = veoFetch([{done: true, error: {code: 3, message: "prompt rejected"}}])
    const out = await events(await handle(req(VEO3, "a cat"), ctxFor(fetch)))
    expect(out.at(-1).candidates[0].finishReason).toBe("OTHER")
    expect(textOf(out.at(-1))).toBe("Error: Veo failed: prompt rejected")
  })

  test("stops polling once veoTimeoutSec is exhausted, reported in the stream", async () => {
    const {fetch, calls} = veoFetch([{done: false}])
    const out = await events(await handle(req(VEO3, "a cat"), ctxFor(fetch, {veoTimeoutSec: 0})))
    expect(out.at(-1).candidates[0].finishReason).toBe("OTHER")
    expect(textOf(out.at(-1))).toBe("Error: Veo did not finish within 0s")
    expect(pollCalls(calls)).toHaveLength(0)
  })
})

describe("submit failures are HTTP errors before the stream opens", () => {
  test("a rejected submit returns the upstream status with the Google error message and no stream", async () => {
    const {fetch, calls} = veoFetch([], {
      submit: () => Response.json({error: {code: 400, message: "Prompt rejected", status: "INVALID_ARGUMENT"}}, {status: 400}),
    })
    const res = await handle(req(VEO3, "a cat"), ctxFor(fetch))
    expect(res.status).toBe(400)
    expect(res.headers.get("content-type")).not.toBe("text/event-stream")
    const body = await res.json()
    expect(body.error.code).toBe(400)
    expect(body.error.message).toBe("Prompt rejected")
    expect(pollCalls(calls)).toHaveLength(0)
  })

  test("a non-JSON upstream body is passed through as the message", async () => {
    const {fetch} = veoFetch([], {submit: () => new Response("invalid prompt", {status: 400})})
    const res = await handle(req(VEO3, "a cat"), ctxFor(fetch))
    expect(res.status).toBe(400)
    expect((await res.json()).error.message).toBe("invalid prompt")
  })

  test("when every region is busy, the last failure's status and message are returned", async () => {
    const {fetch} = veoFetch([], {submit: () => new Response("busy", {status: 503})})
    const res = await handle(req(VEO2, "a cat"), ctxFor(fetch))
    expect(res.status).toBe(503)
    expect((await res.json()).error.message).toBe("busy")
  })

  test("an unreachable upstream is a 502", async () => {
    const {fetch} = veoFetch([], {
      submit: () => {
        throw new TypeError("fetch failed")
      },
    })
    const res = await handle(req(VEO3, "a cat"), ctxFor(fetch))
    expect(res.status).toBe(502)
    expect((await res.json()).error.message).toContain("fetch failed")
  })

  test("a 2xx submit without an operation name is a 502", async () => {
    const {fetch, calls} = veoFetch([], {submit: () => Response.json({})})
    const res = await handle(req(VEO3, "a cat"), ctxFor(fetch))
    expect(res.status).toBe(502)
    expect((await res.json()).error.message).toBe("Veo submit returned no operation name")
    expect(pollCalls(calls)).toHaveLength(0)
  })
})

describe("region failover", () => {
  test("fails over on submit only and polls the region that accepted the job", async () => {
    const {fetch, calls} = veoFetch([{done: true, response: {videos: [{bytesBase64Encoded: MP4, mimeType: "video/mp4"}]}}])
    const submitted = []
    const routed = async (url, init) => {
      if (!url.endsWith(":predictLongRunning")) return fetch(url, init)
      submitted.push(url)
      return url.includes("us-central1") ? new Response("busy", {status: 503}) : Response.json({name: OP})
    }
    await events(await handle(req(VEO2, "a cat"), ctxFor(routed)))
    expect(submitted.map((u) => u.split("/")[2])).toEqual(["us-central1-aiplatform.googleapis.com", "europe-west3-aiplatform.googleapis.com"])
    const poll = calls.find((c) => c.url.endsWith(":fetchPredictOperation"))
    expect(poll.url.split("/")[2]).toBe("europe-west3-aiplatform.googleapis.com")
  })
})
