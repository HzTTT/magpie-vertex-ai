import { describe, expect, test } from "bun:test"
import { regionsFor, retryable, withRegions } from "../lib/failover.mjs"

describe("regionsFor", () => {
  const entry = { regions: ["global", "us-central1", "europe-west4"] }

  test("a pinned region is the only one tried", () => {
    expect(regionsFor(entry, { location: "auto" }, "europe-west1")).toEqual(["europe-west1"])
  })

  test("the preferred region goes first when the model serves it", () => {
    expect(regionsFor(entry, { location: "europe-west4" })).toEqual(["europe-west4", "global", "us-central1"])
  })

  test("catalog order when the preference is auto, or the model does not serve it", () => {
    expect(regionsFor(entry, { location: "auto" })).toEqual(["global", "us-central1", "europe-west4"])
    expect(regionsFor(entry, { location: "asia-east1" })).toEqual(["global", "us-central1", "europe-west4"])
  })
})

describe("retryable", () => {
  test.each([404, 429, 500, 502, 503, 504])("status %i is retryable", (status) => {
    expect(retryable(status, "")).toBe(true)
  })

  test("400 and 403 are retryable only when the text names the region", () => {
    expect(retryable(400, "Publisher Model is not available in this region")).toBe(true)
    expect(retryable(400, "The model is only available via the global endpoint")).toBe(true)
    expect(retryable(403, "Quota exceeded: RESOURCE_EXHAUSTED")).toBe(true)
    expect(retryable(400, "Invalid JSON payload")).toBe(false)
    expect(retryable(403, "Permission denied on resource")).toBe(false)
  })

  test("other statuses are not retryable, whatever the text", () => {
    expect(retryable(401, "not available in")).toBe(false)
  })
})

// replies builds an attempt that answers each region with the next scripted reply and records the regions asked.
function replies(...list) {
  const seen = []
  const attempt = async (loc) => {
    seen.push(loc)
    const r = list.shift()
    return new Response(r.body, { status: r.status, headers: r.type ? { "content-type": r.type } : {} })
  }
  return { attempt, seen }
}

describe("withRegions", () => {
  test("fails over 429 then 404 to a 200 and returns that success", async () => {
    const { attempt, seen } = replies({ status: 429, body: "quota" }, { status: 404, body: "no model" }, { status: 200, body: "ok" })
    const res = await withRegions(["global", "us-central1", "europe-west4"], attempt)
    expect(seen).toEqual(["global", "us-central1", "europe-west4"])
    expect(res.status).toBe(200)
    expect(await res.text()).toBe("ok")
  })

  test("when every region fails, returns the last failure's status, body and content type", async () => {
    const { attempt, seen } = replies({ status: 429, body: "quota", type: "application/json" }, { status: 404, body: "not found", type: "text/plain" })
    const res = await withRegions(["global", "us-central1"], attempt)
    expect(seen).toEqual(["global", "us-central1"])
    expect(res.status).toBe(404)
    expect(await res.text()).toBe("not found")
    expect(res.headers.get("content-type")).toBe("text/plain")
  })

  test("a failure that another region would repeat is returned at once", async () => {
    const { attempt, seen } = replies({ status: 400, body: "Invalid JSON payload", type: "application/json" }, { status: 200, body: "never" })
    const res = await withRegions(["global", "us-central1"], attempt)
    expect(seen).toEqual(["global"])
    expect(res.status).toBe(400)
    expect(await res.text()).toBe("Invalid JSON payload")
  })

  test("no regions is an error", async () => {
    await expect(withRegions([], async () => new Response("x"))).rejects.toThrow("no region serves this model")
  })
})
