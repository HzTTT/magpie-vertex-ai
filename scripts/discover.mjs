// Probes which Vertex models a project can call and writes the result to the magpie config dir.
// Usage: bun scripts/discover.mjs [--credentials path] [--project id] [--dir magpie-config-dir]
import { homedir } from "node:os"
import { join } from "node:path"
import { accessToken, projectOf, readCredentials } from "../lib/auth.mjs"
import { discoveredFile } from "../lib/catalog.mjs"
import { discover } from "../lib/discover.mjs"

const USAGE = `usage: bun scripts/discover.mjs [--credentials path] [--project id] [--dir magpie-config-dir]
  --credentials  service account or authorized_user JSON (default: $GOOGLE_APPLICATION_CREDENTIALS)
  --project      GCP project id (default: project_id from the credentials)
  --dir          magpie config dir (default: $XDG_CONFIG_HOME/magpie or ~/.config/magpie)`

function parseArgs(argv) {
  const opts = {}
  for (let i = 0; i < argv.length; i++) {
    const flag = argv[i]
    const takeValue = () => {
      const value = argv[++i]
      if (value === undefined) throw new Error(`${flag} needs a value`)
      return value
    }
    if (flag === "--credentials") opts.credentials = takeValue()
    else if (flag === "--project") opts.project = takeValue()
    else if (flag === "--dir") opts.dir = takeValue()
    else if (flag === "--help" || flag === "-h") opts.help = true
    else throw new Error(`unknown option ${flag}\n${USAGE}`)
  }
  return opts
}

function defaultDir() {
  const base = process.env.XDG_CONFIG_HOME || join(homedir(), ".config")
  return join(base, "magpie")
}

// At most ~20 stderr lines: each phase ticks its own ~5% buckets, sharing the budget.
const STEPS = { list: 2, probe: 16, gate: 1 }

function progress() {
  const seen = new Map()
  return (e) => {
    const steps = STEPS[e.phase] ?? 4
    const bucket = Math.min(steps, Math.floor((e.done / e.total) * steps))
    if (bucket <= Math.max(seen.get(e.phase) ?? -1, 0)) return
    seen.set(e.phase, bucket)
    process.stderr.write(`${e.phase} ${e.done}/${e.total}\n`)
  }
}

function printSummary(data, file) {
  const idWidth = Math.max(...data.models.map((m) => m.id.length), 5)
  console.log(`${"model".padEnd(idWidth)}  ${"family".padEnd(13)}  regions  first regions`)
  const families = new Map()
  for (const m of data.models) {
    families.set(m.family, (families.get(m.family) ?? 0) + 1)
    console.log(`${m.id.padEnd(idWidth)}  ${m.family.padEnd(13)}  ${String(m.regions.length).padStart(7)}  ${m.regions.slice(0, 4).join(", ")}`)
  }
  const reasons = new Map()
  for (const u of data.unavailable) reasons.set(u.reason, (reasons.get(u.reason) ?? 0) + 1)
  const reasonText = [...reasons].map(([reason, count]) => `${reason}: ${count}`).join(", ")
  const familyText = [...families].map(([family, count]) => `${family}: ${count}`).join(", ")
  console.log(`\n${data.models.length} callable, ${data.unavailable.length} unavailable${reasonText ? ` (${reasonText})` : ""}`)
  console.log(`by family: ${familyText || "none"}`)
  console.log(`written: ${file}`)
}

async function main() {
  const opts = parseArgs(process.argv.slice(2))
  if (opts.help) {
    console.log(USAGE)
    return
  }
  const credentials = opts.credentials ?? process.env.GOOGLE_APPLICATION_CREDENTIALS
  if (!credentials) throw new Error(`pass --credentials <file> (or set GOOGLE_APPLICATION_CREDENTIALS)\n${USAGE}`)
  const dir = opts.dir ?? defaultDir()

  const { creds } = await readCredentials(credentials)
  const project = opts.project ?? projectOf(creds)
  if (!project) throw new Error("the credentials have no project; pass --project <id>")

  const ctx = {
    project,
    token: () => accessToken(creds),
    fetch: globalThis.fetch,
    log: (msg) => process.stderr.write(`${msg}\n`),
  }
  const data = await discover(ctx, { dir, onProgress: progress() })
  printSummary(data, discoveredFile(dir, project))
}

main().catch((err) => {
  console.error(err.message)
  process.exit(1)
})
