import { beforeAll, beforeEach, describe, expect, test } from "bun:test"
import { mkdtemp, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { accessToken, authorizeAdc, authorizeCredentials, checkCredentials, clearTokenCache, identity, parseCredentials, readCredentials, refreshSignIn, tokenOf } from "../lib/auth.mjs"

const enc = new TextEncoder()
const TOKEN_URI = "https://oauth2.googleapis.com/token"
const JWT_GRANT = "urn:ietf:params:oauth:grant-type:jwt-bearer"

// A throwaway RSA key, generated per run; nothing here touches real credentials.
let key

beforeAll(async () => {
  const pair = await crypto.subtle.generateKey(
    { name: "RSASSA-PKCS1-v1_5", modulusLength: 2048, publicExponent: new Uint8Array([1, 0, 1]), hash: "SHA-256" },
    true,
    ["sign", "verify"],
  )
  const der = Buffer.from(await crypto.subtle.exportKey("pkcs8", pair.privateKey))
  const body = der.toString("base64").match(/.{1,64}/g).join("\n")
  key = { publicKey: pair.publicKey, pem: `-----BEGIN PRIVATE KEY-----\n${body}\n-----END PRIVATE KEY-----\n` }
})

beforeEach(() => clearTokenCache())

function serviceAccount(email = "vertex@demo.iam.gserviceaccount.com") {
  return {
    type: "service_account",
    project_id: "demo",
    private_key_id: "kid-1",
    private_key: key.pem,
    client_email: email,
    token_uri: TOKEN_URI,
  }
}

// fakeTokenServer answers each token request with the next reply; the last one repeats.
function fakeTokenServer(...replies) {
  const calls = []
  const fetch = async (url, init) => {
    calls.push({ url, params: new URLSearchParams(String(init.body)) })
    const r = replies.length > 1 ? replies.shift() : replies[0]
    return new Response(JSON.stringify(r.body), { status: r.status ?? 200, headers: { "content-type": "application/json" } })
  }
  return { fetch, calls }
}

// withFetch runs fn with the global fetch replaced: the sign-in callbacks use it by default.
async function withFetch(fetch, fn) {
  const real = globalThis.fetch
  globalThis.fetch = fetch
  try {
    return await fn()
  } finally {
    globalThis.fetch = real
  }
}

// withEnv sets environment variables for fn, then restores them.
async function withEnv(vars, fn) {
  const saved = Object.fromEntries(Object.keys(vars).map((k) => [k, process.env[k]]))
  Object.assign(process.env, vars)
  for (const k of Object.keys(vars)) if (vars[k] === undefined) delete process.env[k]
  try {
    return await fn()
  } finally {
    for (const [k, v] of Object.entries(saved)) {
      if (v === undefined) delete process.env[k]
      else process.env[k] = v
    }
  }
}

describe("service account JWT", () => {
  test("signs an RS256 assertion whose claims and signature check out", async () => {
    const { fetch, calls } = fakeTokenServer({ body: { access_token: "tok-1", expires_in: 3599 } })
    const before = Math.floor(Date.now() / 1000)
    expect(await accessToken(serviceAccount(), { fetch })).toBe("tok-1")

    expect(calls.length).toBe(1)
    expect(calls[0].url).toBe(TOKEN_URI)
    expect(calls[0].params.get("grant_type")).toBe(JWT_GRANT)
    const [h, c, s] = calls[0].params.get("assertion").split(".")
    expect(JSON.parse(Buffer.from(h, "base64url").toString())).toEqual({ alg: "RS256", typ: "JWT" })
    const claims = JSON.parse(Buffer.from(c, "base64url").toString())
    expect(claims.iss).toBe("vertex@demo.iam.gserviceaccount.com")
    expect(claims.scope).toBe("https://www.googleapis.com/auth/cloud-platform")
    expect(claims.aud).toBe(TOKEN_URI)
    expect(claims.iat).toBeGreaterThanOrEqual(before)
    expect(claims.exp - claims.iat).toBe(3600)
    const valid = await crypto.subtle.verify({ name: "RSASSA-PKCS1-v1_5" }, key.publicKey, Buffer.from(s, "base64url"), enc.encode(`${h}.${c}`))
    expect(valid).toBe(true)
  })

  test("reuses the token until five minutes before it expires", async () => {
    const { fetch, calls } = fakeTokenServer({ body: { access_token: "tok-1", expires_in: 3599 } })
    const creds = serviceAccount()
    expect(await accessToken(creds, { fetch })).toBe("tok-1")
    expect(await accessToken(creds, { fetch })).toBe("tok-1")
    expect(calls.length).toBe(1)
  })

  test("refreshes a token that is already inside the refresh window", async () => {
    const { fetch, calls } = fakeTokenServer({ body: { access_token: "short", expires_in: 60 } }, { body: { access_token: "fresh", expires_in: 3599 } })
    const creds = serviceAccount()
    expect(await accessToken(creds, { fetch })).toBe("short")
    expect(await accessToken(creds, { fetch })).toBe("fresh")
    expect(calls.length).toBe(2)
  })

  test("single-flights concurrent requests for one identity", async () => {
    const { fetch, calls } = fakeTokenServer({ body: { access_token: "tok-1", expires_in: 3599 } })
    const creds = serviceAccount()
    const tokens = await Promise.all([accessToken(creds, { fetch }), accessToken(creds, { fetch }), accessToken(creds, { fetch })])
    expect(tokens).toEqual(["tok-1", "tok-1", "tok-1"])
    expect(calls.length).toBe(1)
  })

  test("keys the cache by identity", async () => {
    const { fetch, calls } = fakeTokenServer({ body: { access_token: "tok", expires_in: 3599 } })
    await accessToken(serviceAccount("a@demo.iam.gserviceaccount.com"), { fetch })
    await accessToken(serviceAccount("b@demo.iam.gserviceaccount.com"), { fetch })
    expect(calls.length).toBe(2)
  })

  test.each(["invalid_grant", "invalid_client", "unauthorized_client"])("%s is a sign-in that expired, with Google's message", async (code) => {
    const { fetch } = fakeTokenServer({ status: 400, body: { error: code, error_description: "Invalid JWT Signature." } })
    const err = await accessToken(serviceAccount(), { fetch }).catch((e) => e)
    expect(err).toBeInstanceOf(Error)
    expect(err.signIn).toBe("expired")
    expect(err.message).toContain("Invalid JWT Signature.")
  })

  test("other token failures are not a sign-in that expired", async () => {
    const { fetch } = fakeTokenServer({ status: 503, body: { error: "temporarily_unavailable" } })
    const err = await accessToken(serviceAccount(), { fetch }).catch((e) => e)
    expect(err.message).toContain("temporarily_unavailable")
    expect(err.signIn).toBeUndefined()
  })

  test("a private key that cannot be imported is an expired sign-in, and Google is not asked", async () => {
    const { fetch, calls } = fakeTokenServer({ body: { access_token: "never", expires_in: 3599 } })
    const creds = { ...serviceAccount(), private_key: "-----BEGIN PRIVATE KEY-----\nAAAA\n-----END PRIVATE KEY-----\n" }
    const err = await accessToken(creds, { fetch }).catch((e) => e)
    expect(err.signIn).toBe("expired")
    expect(calls.length).toBe(0)
  })
})

describe("authorized_user credentials", () => {
  test("use the refresh_token grant against the default token endpoint", async () => {
    const { fetch, calls } = fakeTokenServer({ body: { access_token: "adc-tok", expires_in: 3599 } })
    const creds = { type: "authorized_user", client_id: "cid", client_secret: "sec", refresh_token: "rt-1" }
    expect(await accessToken(creds, { fetch })).toBe("adc-tok")
    expect(calls[0].url).toBe(TOKEN_URI)
    expect(calls[0].params.get("grant_type")).toBe("refresh_token")
    expect(calls[0].params.get("refresh_token")).toBe("rt-1")
    expect(calls[0].params.has("assertion")).toBe(false)
  })
})

describe("credentials", () => {
  test("parseCredentials accepts service account and authorized_user JSON", () => {
    expect(parseCredentials(JSON.stringify(serviceAccount())).type).toBe("service_account")
    expect(parseCredentials(JSON.stringify({ type: "authorized_user", client_id: "c", client_secret: "s", refresh_token: "r" })).type).toBe("authorized_user")
  })

  test("parseCredentials rejects anything else with a readable message", () => {
    expect(() => parseCredentials("not json")).toThrow("not JSON")
    expect(() => parseCredentials(JSON.stringify({ type: "external_account" }))).toThrow('Unsupported credentials type "external_account"')
    expect(() => parseCredentials(JSON.stringify({ type: "service_account", client_email: "a@b.c" }))).toThrow("lack private_key")
  })

  test("identity is the service account email, or user:<client_id>", () => {
    expect(identity(serviceAccount("x@y.iam.gserviceaccount.com"))).toBe("x@y.iam.gserviceaccount.com")
    expect(identity({ type: "authorized_user", client_id: "cid", client_secret: "s", refresh_token: "r" })).toBe("user:cid")
  })

  test("readCredentials takes JSON text, or a file path with $VAR expanded", async () => {
    const dir = await mkdtemp(join(tmpdir(), "vertex-auth-"))
    try {
      const text = JSON.stringify(serviceAccount())
      const fromJson = await readCredentials(`  ${text}\n`)
      expect(fromJson.creds.client_email).toBe("vertex@demo.iam.gserviceaccount.com")
      expect(fromJson.text).toBe(text)

      await writeFile(join(dir, "sa.json"), text)
      await withEnv({ MAGPIE_VERTEX_TEST_DIR: dir }, async () => {
        const fromPath = await readCredentials("$MAGPIE_VERTEX_TEST_DIR/sa.json")
        expect(fromPath.text).toBe(text)
      })

      await expect(readCredentials(join(dir, "missing.json"))).rejects.toThrow("Cannot read credentials file")
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("checkCredentials passes JSON and readable files, and names what it cannot use", async () => {
    const dir = await mkdtemp(join(tmpdir(), "vertex-check-"))
    try {
      const text = JSON.stringify(serviceAccount())
      expect(checkCredentials(text)).toBeNull()
      await writeFile(join(dir, "sa.json"), text)
      expect(checkCredentials(join(dir, "sa.json"))).toBeNull()
      expect(checkCredentials(join(dir, "missing.json"))).toContain("Cannot read")
      expect(checkCredentials('{"type":"external_account"}')).toContain('Unsupported credentials type "external_account"')
      expect(checkCredentials('{"type": ')).toContain("not JSON")
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("checkCredentials with nothing typed uses GOOGLE_APPLICATION_CREDENTIALS, and says so when that is unset too", async () => {
    await withEnv({ GOOGLE_APPLICATION_CREDENTIALS: undefined }, async () => {
      expect(checkCredentials("")).toContain("GOOGLE_APPLICATION_CREDENTIALS")
    })
    await withEnv({ GOOGLE_APPLICATION_CREDENTIALS: JSON.stringify(serviceAccount()) }, async () => {
      expect(checkCredentials("")).toBeNull()
    })
  })
})

describe("service account sign-in", () => {
  test("authorize has nothing to open: the sign-in waits for its callback, which asks Google", async () => {
    const { fetch, calls } = fakeTokenServer({ body: { access_token: "tok", expires_in: 3599 } })
    await withFetch(fetch, async () => {
      const sign = authorizeCredentials({ credentials: JSON.stringify(serviceAccount()), project: "", location: "auto" })
      expect(sign.url).toBe("")
      expect(sign.method).toBe("auto")
      expect(typeof sign.callback).toBe("function")
      expect(calls.length).toBe(0)
    })
  })

  test("a successful callback gives the credentials as refresh, a token with its expiry, and the account's identity", async () => {
    const { fetch, calls } = fakeTokenServer({ body: { access_token: "tok-1", expires_in: 3599 } })
    const text = JSON.stringify(serviceAccount())
    const result = await withFetch(fetch, () => authorizeCredentials({ credentials: text, project: "", location: "us-central1" }).callback())
    expect(result).toEqual({
      type: "success",
      refresh: text,
      access: "tok-1",
      expires: expect.any(Number),
      accountId: "vertex@demo.iam.gserviceaccount.com",
      project: "demo",
      location: "us-central1",
    })
    expect(result.expires).toBeGreaterThan(Date.now() + 3000 * 1000)
    expect(calls.length).toBe(1)
  })

  test("a typed project wins over the one in the credentials", async () => {
    const { fetch } = fakeTokenServer({ body: { access_token: "tok", expires_in: 3599 } })
    const result = await withFetch(fetch, () => authorizeCredentials({ credentials: JSON.stringify(serviceAccount()), project: "other-project" }).callback())
    expect(result.project).toBe("other-project")
    expect(result.location).toBe("auto")
  })

  test("a refused grant is a failed callback carrying Google's message", async () => {
    const { fetch } = fakeTokenServer({ status: 400, body: { error: "invalid_grant", error_description: "Invalid JWT Signature." } })
    const result = await withFetch(fetch, () => authorizeCredentials({ credentials: JSON.stringify(serviceAccount()) }).callback())
    expect(result).toEqual({ type: "failed", error: expect.stringContaining("Invalid JWT Signature.") })
  })

  test("nothing typed and no GOOGLE_APPLICATION_CREDENTIALS is a failed callback", async () => {
    await withEnv({ GOOGLE_APPLICATION_CREDENTIALS: undefined }, async () => {
      const result = await authorizeCredentials({}).callback()
      expect(result).toEqual({ type: "failed", error: expect.stringContaining("GOOGLE_APPLICATION_CREDENTIALS") })
    })
  })
})

describe("application default credentials sign-in", () => {
  test("reads the file named by GOOGLE_APPLICATION_CREDENTIALS and stores it as an authorized_user identity", async () => {
    const dir = await mkdtemp(join(tmpdir(), "vertex-adc-"))
    try {
      const adc = { type: "authorized_user", client_id: "cid", client_secret: "sec", refresh_token: "rt-1" }
      const file = join(dir, "adc.json")
      await writeFile(file, JSON.stringify(adc))
      const { fetch } = fakeTokenServer({ body: { access_token: "adc-tok", expires_in: 3599 } })
      const result = await withEnv({ GOOGLE_APPLICATION_CREDENTIALS: file }, () => withFetch(fetch, () => authorizeAdc({ project: "demo", location: "europe-west4" }).callback()))
      expect(result).toEqual({
        type: "success",
        refresh: JSON.stringify(adc),
        access: "adc-tok",
        expires: expect.any(Number),
        accountId: "user:cid",
        project: "demo",
        location: "europe-west4",
      })
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("authorized_user credentials without a project need one typed", async () => {
    const dir = await mkdtemp(join(tmpdir(), "vertex-adc-"))
    try {
      const file = join(dir, "adc.json")
      await writeFile(file, JSON.stringify({ type: "authorized_user", client_id: "cid", client_secret: "sec", refresh_token: "rt-1" }))
      const { fetch } = fakeTokenServer({ body: { access_token: "adc-tok", expires_in: 3599 } })
      const result = await withEnv({ GOOGLE_APPLICATION_CREDENTIALS: file, GOOGLE_CLOUD_PROJECT: undefined }, () => withFetch(fetch, () => authorizeAdc({}).callback()))
      expect(result).toEqual({ type: "failed", error: expect.stringContaining("No project") })
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })
})

describe("refresh and the stored token", () => {
  test("refreshSignIn mints a new token even when the cache holds a good one", async () => {
    const { fetch, calls } = fakeTokenServer({ body: { access_token: "old", expires_in: 3599 } }, { body: { access_token: "new", expires_in: 3599 } })
    const auth = { type: "oauth", refresh: JSON.stringify(serviceAccount()) }
    expect(await accessToken(parseCredentials(auth.refresh), { fetch })).toBe("old")
    const renewed = await refreshSignIn(auth, { fetch })
    expect(renewed.access).toBe("new")
    expect(renewed.expires).toBeGreaterThan(Date.now())
    expect(calls.length).toBe(2)
  })

  test("refreshSignIn turns a refused grant into a sign-in that expired", async () => {
    const { fetch } = fakeTokenServer({ status: 400, body: { error: "invalid_grant", error_description: "Invalid JWT Signature." } })
    const err = await refreshSignIn({ type: "oauth", refresh: JSON.stringify(serviceAccount()) }, { fetch }).catch((e) => e)
    expect(err.signIn).toBe("expired")
  })

  test("refreshSignIn with credentials that cannot be read is a sign-in that expired", async () => {
    const err = await refreshSignIn({ type: "oauth", refresh: "garbage" }).catch((e) => e)
    expect(err.signIn).toBe("expired")
  })

  test("tokenOf uses the stored access token while more than five minutes are left", async () => {
    const { fetch, calls } = fakeTokenServer({ body: { access_token: "minted", expires_in: 3599 } })
    const auth = { type: "oauth", refresh: JSON.stringify(serviceAccount()), access: "stored", expires: Date.now() + 3600 * 1000 }
    expect(await tokenOf(auth, { fetch })).toBe("stored")
    expect(calls.length).toBe(0)
  })

  test("tokenOf mints from the credentials once the stored token is inside the refresh window", async () => {
    const { fetch, calls } = fakeTokenServer({ body: { access_token: "minted", expires_in: 3599 } })
    const auth = { type: "oauth", refresh: JSON.stringify(serviceAccount()), access: "stored", expires: Date.now() + 60 * 1000 }
    expect(await tokenOf(auth, { fetch })).toBe("minted")
    expect(calls.length).toBe(1)
  })
})
