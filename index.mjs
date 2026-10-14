// magpie provider plugin for Google Vertex AI: every callable model of the
// account's GCP project, reached through the plugin fetch (lib/route.mjs).
import { stat } from "node:fs/promises"
import { join } from "node:path"
import { authorizeAdc, authorizeCredentials, buildCtx, checkCredentials, accessToken, identity, parseCredentials, projectOf, readCredentials, refreshSignIn, tokenOf } from "./lib/auth.mjs"
import { CATALOG, configModels, loadDiscovered, mergeDiscovered, modelsFor } from "./lib/catalog.mjs"
import { discover, isStale } from "./lib/discover.mjs"
import { route } from "./lib/route.mjs"
import { BASE, REGIONS } from "./lib/regions.mjs"

const PROVIDER = "vertex-ai"
const SERVICE = "magpie-vertex-ai"

// The credentials source is asked as a prompt: magpie hands authorize only the
// prompt answers, never a typed key. validate checks it before the sign-in.
const CREDENTIALS = {
  type: "text",
  key: "credentials",
  message: "Credentials: a JSON file path, or the JSON itself on one line (empty: GOOGLE_APPLICATION_CREDENTIALS)",
  placeholder: "~/.config/gcp/vertex-sa.json",
  validate: checkCredentials,
}
const PROJECT = { type: "text", key: "project", message: "GCP project id", placeholder: "empty: the project in the credentials" }
const LOCATION = {
  type: "select",
  key: "location",
  message: "Region",
  options: [
    { label: "auto", value: "auto", hint: "recommended: the first region that serves the model" },
    ...REGIONS.map((region) => ({ label: region, value: region })),
  ],
}

const discovered = new Map() // catalog file → {mtime, entries}
const started = new Set() // projects a discovery was started for in this process

export const VertexAIPlugin = async (input, options = {}) => {
  const dir = input.directory
  const log = logger(input)
  const ctxFor = (getAuth, project, location) =>
    buildCtx({
      getAuth,
      project,
      location,
      options,
      directory: dir,
      log: (msg) => log("info", msg),
    })

  return {
    config: async (cfg) => {
      cfg.provider ??= {}
      cfg.provider[PROVIDER] ??= {
        name: "Google Vertex AI",
        npm: "@ai-sdk/openai-compatible",
        api: BASE,
        models: configModels(),
      }
    },

    auth: {
      provider: PROVIDER,
      methods: [
        { type: "oauth", label: "Service account / credentials JSON", prompts: [CREDENTIALS, PROJECT, LOCATION], authorize: authorizeCredentials },
        { type: "oauth", label: "Application Default Credentials", prompts: [PROJECT, LOCATION], authorize: authorizeAdc },
      ],
      // magpie renews the access token before it expires, through this hook.
      refresh: (auth) => refreshSignIn(auth),
      loader: async (getAuth) => {
        const auth = await getAuth()
        if (auth?.type !== "oauth") return {}
        const creds = parseCredentials(auth.refresh)
        const project = projectOf(creds, auth.project)
        if (!project) throw new Error("No GCP project for this account: sign in again and name one")
        const ctx = ctxFor(getAuth, project, auth.location || options.location)
        return {
          baseURL: BASE,
          fetch: async (url, init) => route(url, init, ctx, { entries: await entriesFor(dir, project, log) }),
        }
      },
    },

    provider: {
      id: PROVIDER,
      models: async (given, { auth } = {}) => {
        if (auth?.type !== "oauth") return given.models
        try {
          const creds = parseCredentials(auth.refresh)
          const project = projectOf(creds, auth.project)
          if (!project) return given.models
          const ctx = ctxFor(async () => auth, project, auth.location || options.location)
          // A sign-in the vendor refuses throws here with signIn "expired".
          await ctx.token()
          const ttlHours = options.discoverTtlHours ?? 24
          if (options.autoDiscover !== false && (await missingOrStale(discoveredPath(dir, project), ttlHours))) {
            discoverOnce(ctx, dir, log)
          }
          return modelsFor({ entries: await entriesFor(dir, project, log), options })
        } catch (e) {
          if (e?.signIn) throw e
          log("error", `models: ${e?.message ?? e}`)
          return given.models
        }
      },
    },
  }
}

function discoveredPath(dir, project) {
  return join(dir, "vertex-ai", `catalog-${project}.json`)
}

// The project's catalog: the discovered one, reloaded when its file changes.
async function entriesFor(dir, project, log) {
  const file = discoveredPath(dir, project)
  const st = await stat(file).catch(() => null)
  const mtime = st?.mtimeMs ?? 0
  const hit = discovered.get(file)
  if (hit && hit.mtime === mtime) return hit.entries
  let entries = CATALOG
  if (st) {
    try {
      const d = await loadDiscovered(dir, project)
      if (d) entries = mergeDiscovered(d)
    } catch (e) {
      log("error", `${file}: ${e?.message ?? e}`)
    }
  }
  discovered.set(file, { mtime, entries })
  return entries
}

async function missingOrStale(file, ttlHours) {
  const st = await stat(file).catch(() => null)
  return !st || isStale(file, ttlHours)
}

function discoverOnce(ctx, dir, log) {
  if (started.has(ctx.project)) return
  started.add(ctx.project)
  discover(ctx, { dir }).catch((e) => log("error", `catalog discovery for ${ctx.project} failed: ${e?.message ?? e}`))
}

// Plugin logs go to OpenCode's app.log when it is there, else to stderr (magpie logs it).
function logger(input) {
  return (level, message) => {
    const app = input?.client?.app
    if (typeof app?.log !== "function") return console.error(`${SERVICE} ${level}: ${message}`)
    Promise.resolve()
      .then(() => app.log({ body: { service: SERVICE, level, message } }))
      .catch(() => {})
  }
}

export const _internal = {
  route,
  buildCtx,
  entriesFor,
  discoveredPath,
  parseCredentials,
  readCredentials,
  checkCredentials,
  accessToken,
  tokenOf,
  refreshSignIn,
  identity,
  authorizeCredentials,
  authorizeAdc,
  projectOf,
}
