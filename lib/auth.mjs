import { readFileSync } from "node:fs"
import { readFile } from "node:fs/promises"
import { homedir } from "node:os"
import { join, resolve } from "node:path"

const SCOPE = "https://www.googleapis.com/auth/cloud-platform"
const DEFAULT_TOKEN_URI = "https://oauth2.googleapis.com/token"
const JWT_GRANT = "urn:ietf:params:oauth:grant-type:jwt-bearer"
// Tokens are renewed this long before they expire; magpie renews the stored access token on the same lead.
const REFRESH_LEAD_MS = 5 * 60 * 1000
// OAuth errors that mean the credentials are no longer good, not that Google is down.
const REFUSED = new Set(["invalid_grant", "invalid_client", "unauthorized_client"])

const tokens = new Map() // cache key → {token, expiresAt}
const minting = new Map() // cache key → the token request under way (single-flight)

export function parseCredentials(text) {
  let creds
  try {
    creds = JSON.parse(String(text))
  } catch {
    throw new Error("Credentials are not JSON: give a service account key or an authorized_user (gcloud ADC) file")
  }
  const type = creds && typeof creds === "object" ? creds.type : undefined
  const required = type === "service_account" ? ["client_email", "private_key"] : type === "authorized_user" ? ["client_id", "client_secret", "refresh_token"] : null
  if (!required) throw new Error(`Unsupported credentials type ${JSON.stringify(type ?? null)}: use a service account key or an authorized_user (gcloud ADC) file`)
  const missing = required.filter((k) => !creds[k])
  if (missing.length) throw new Error(`${type} credentials lack ${missing.join(", ")}`)
  return creds
}

// The credentials source: what was typed, else GOOGLE_APPLICATION_CREDENTIALS.
function sourceOf(typed) {
  return String(typed ?? "").trim() || process.env.GOOGLE_APPLICATION_CREDENTIALS || ""
}

// Checks the typed credentials prompt without the network (magpie's prompt validate):
// the source must be readable and parse as credentials.
export function checkCredentials(value) {
  const raw = sourceOf(value)
  if (!raw) return "Give a credentials file path or the JSON, or set GOOGLE_APPLICATION_CREDENTIALS"
  let text = raw
  if (!raw.startsWith("{")) {
    try {
      text = readFileSync(expandPath(raw), "utf8")
    } catch (e) {
      return `Cannot read ${raw}: ${e.code ?? e.message}`
    }
  }
  try {
    parseCredentials(text)
  } catch (e) {
    return e.message
  }
  return null
}

// JSON text, or a path (~ and $VARS expanded) to a file holding it.
export async function readCredentials(pathOrJson) {
  const raw = String(pathOrJson ?? "").trim()
  if (!raw) throw new Error("No credentials given")
  let text = raw
  if (!raw.startsWith("{")) {
    try {
      text = (await readFile(expandPath(raw), "utf8")).trim()
    } catch (e) {
      throw new Error(`Cannot read credentials file ${raw}: ${e.code ?? e.message}`)
    }
  }
  return { creds: parseCredentials(text), text }
}

function expandPath(p) {
  return p.replace(/^~(?=\/|$)/, homedir()).replace(/\$\{(\w+)\}|\$(\w+)/g, (m, a, b) => process.env[a ?? b] ?? m)
}

export function identity(creds) {
  return creds.type === "service_account" ? creds.client_email : `user:${creds.client_id}`
}

export function projectOf(creds, preferred) {
  return preferred || creds.project_id || creds.quota_project_id || process.env.GOOGLE_CLOUD_PROJECT || ""
}

// {token, expiresAt} for these credentials: the cached one until the refresh
// window, else a new one. force mints anyway. Concurrent callers share one request.
export function tokenInfo(creds, { fetch = globalThis.fetch, force = false } = {}) {
  const key = cacheKey(creds)
  const hit = tokens.get(key)
  if (!force && hit && hit.expiresAt - Date.now() > REFRESH_LEAD_MS) return Promise.resolve(hit)
  if (!minting.has(key)) {
    minting.set(
      key,
      mint(creds, fetch)
        .then((fresh) => {
          tokens.set(key, fresh)
          return fresh
        })
        .finally(() => minting.delete(key)),
    )
  }
  return minting.get(key)
}

export async function accessToken(creds, opts) {
  return (await tokenInfo(creds, opts)).token
}

// The bearer token for a signed-in account: the stored access token while magpie
// keeps it with more than the refresh lead left, else one minted from its credentials.
export async function tokenOf(auth, { fetch = globalThis.fetch } = {}) {
  if (!auth) throw new Error("this account is signed out")
  if (auth.access && auth.expires - Date.now() > REFRESH_LEAD_MS) return auth.access
  return accessToken(parseCredentials(auth.refresh), { fetch })
}

// magpie's auth.refresh: a new token for the account's credentials, bypassing the
// cache. Credentials the vendor no longer accepts are a sign-in that expired.
export async function refreshSignIn(auth, { fetch = globalThis.fetch } = {}) {
  let creds
  try {
    creds = parseCredentials(auth.refresh)
  } catch (e) {
    throw Object.assign(new Error(`The stored credentials cannot be used: ${e.message}`), { signIn: "expired" })
  }
  const fresh = await tokenInfo(creds, { fetch, force: true })
  return { access: fresh.token, expires: fresh.expiresAt }
}

export function clearTokenCache() {
  tokens.clear()
  minting.clear()
}

// The account context handlers get (see the contract in local://contract.md).
export function buildCtx({ getAuth, project, location = "auto", options = {}, directory, fetch = globalThis.fetch, log = () => {} }) {
  return {
    project,
    location,
    token: async () => tokenOf(await getAuth(), { fetch }),
    fetch,
    mediaDir: resolve(options.mediaDir ?? join(directory, "vertex-ai", "media")),
    options,
    log,
  }
}

// Both methods are oauth-shaped: magpie's sign-in for an api method takes no
// prompt answer as the key, so the credentials come in as a prompt and the
// sign-in has nothing to open. The callback does the work.
export function authorizeCredentials(inputs = {}) {
  return signInWith(inputs, async () => {
    const source = sourceOf(inputs.credentials)
    if (!source) throw new Error("Give a credentials file path or the JSON, or set GOOGLE_APPLICATION_CREDENTIALS")
    return readCredentials(source)
  })
}

export function authorizeAdc(inputs = {}) {
  return signInWith(inputs, async () => {
    const source = process.env.GOOGLE_APPLICATION_CREDENTIALS || join(homedir(), ".config", "gcloud", "application_default_credentials.json")
    try {
      return await readCredentials(source)
    } catch (e) {
      throw new Error(`No Application Default Credentials at ${source} (${e.message}): run gcloud auth application-default login`)
    }
  })
}

function signInWith(inputs, load) {
  return {
    url: "",
    instructions: "Checking the Google credentials…",
    method: "auto",
    // What magpie saves: refresh holds the credentials JSON, access and expires the token.
    async callback() {
      try {
        const { creds, text } = await load()
        const project = projectOf(creds, (inputs.project ?? "").trim())
        if (!project) return { type: "failed", error: "No project: type a project id, or use credentials that name one" }
        const fresh = await tokenInfo(creds, { force: true })
        return {
          type: "success",
          refresh: text,
          access: fresh.token,
          expires: fresh.expiresAt,
          accountId: identity(creds),
          project,
          location: inputs.location || "auto",
        }
      } catch (e) {
        return { type: "failed", error: e?.message ?? String(e) }
      }
    },
  }
}

function cacheKey(creds) {
  return `${identity(creds)}#${creds.private_key_id ?? creds.refresh_token?.slice(-16) ?? ""}`
}

async function mint(creds, fetch) {
  const uri = creds.token_uri || DEFAULT_TOKEN_URI
  const form =
    creds.type === "service_account"
      ? { grant_type: JWT_GRANT, assertion: await signedAssertion(creds, uri) }
      : { grant_type: "refresh_token", client_id: creds.client_id, client_secret: creds.client_secret, refresh_token: creds.refresh_token }
  const res = await fetch(uri, {
    method: "POST",
    headers: { "content-type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams(form).toString(),
  })
  const text = await res.text()
  const json = parseJson(text)
  if (!res.ok || !json?.access_token) {
    const code = json?.error
    const why = json?.error_description || code || text.slice(0, 200) || `HTTP ${res.status}`
    throw Object.assign(new Error(`Google refused the token request (${res.status}): ${why}`), REFUSED.has(code) ? { signIn: "expired" } : {})
  }
  return { token: json.access_token, expiresAt: Date.now() + Number(json.expires_in ?? 3600) * 1000 }
}

async function signedAssertion(creds, aud) {
  const key = await importKey(creds.private_key)
  const now = Math.floor(Date.now() / 1000)
  const b64 = (v) => Buffer.from(JSON.stringify(v)).toString("base64url")
  const input = `${b64({ alg: "RS256", typ: "JWT" })}.${b64({ iss: creds.client_email, scope: SCOPE, aud, iat: now, exp: now + 3600 })}`
  const sig = await crypto.subtle.sign("RSASSA-PKCS1-v1_5", key, new TextEncoder().encode(input))
  return `${input}.${Buffer.from(sig).toString("base64url")}`
}

async function importKey(pem) {
  const der = Buffer.from(String(pem).replace(/-----[^-]+-----|\s/g, ""), "base64")
  try {
    return await crypto.subtle.importKey("pkcs8", der, { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" }, false, ["sign"])
  } catch (e) {
    throw Object.assign(new Error(`The service account's private key cannot be used (${e?.message ?? e}): create a new key`), { signIn: "expired" })
  }
}

function parseJson(text) {
  try {
    return JSON.parse(text)
  } catch {
    return null
  }
}
