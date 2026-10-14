import {modelUrl} from "../regions.mjs"
import {regionsFor, withRegions} from "../failover.mjs"
import {errorResponse} from "../errors.mjs"
import {saveMedia} from "../media.mjs"
import {lastUserTurn} from "../prompt.mjs"
import {chunk, geminiStream} from "../sse.mjs"

const USAGE = {promptTokenCount: 0, candidatesTokenCount: 0, totalTokenCount: 0}

// Lyria only accepts US-English prompts, and its language filter is strict with
// formatting noise. Strip markdown links, URLs, emphasis markers and whitespace
// runs so a plain music description is what reaches the model — never reword it.
function musicPrompt(text) {
  return text
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, "$1") // [label](url) and ![alt](url): keep the label
    .replace(/\b(?:https?|file):\/\/\S+|\bdata:[^\s]*[;,][^\s]*/g, "") // bare URLs and data: URIs
    .replace(/\*\*([^*]+)\*\*/g, "$1") // **bold**
    .replace(/\*([^*\s][^*]*)\*/g, "$1") // *emphasis*
    .replace(/`([^`]+)`/g, "$1") // `code`
    .replace(/\s+/g, " ")
    .trim()
}

// Google rejects fragment prompts with a bare 400 whose text says "Unsupported
// language detected". Keep the status and Google's message, and add how to phrase
// a prompt the filter accepts.
const LANGUAGE_HINT =
  'Lyria takes US-English prompts only, and its language filter is strict with fragment prompts: describe the music in one or two full English sentences (genre, mood, instruments, tempo), e.g. "A calm lo-fi piano loop with soft brushed drums and a relaxed tempo."'

export async function handle(req, ctx) {
  const prompt = musicPrompt(lastUserTurn(req.body).text)
  if (!prompt) return errorResponse("gemini", 400, "Lyria needs a text prompt in the last user turn")
  const {entry, pinned} = req
  const instance = {prompt}
  if (ctx.options.lyria?.negativePrompt) instance.negative_prompt = ctx.options.lyria.negativePrompt

  // Predict before the stream opens, so an upstream failure is an HTTP error magpie can fail over on.
  let res
  try {
    res = await withRegions(
      regionsFor(entry, ctx, pinned),
      (loc) => post(ctx, modelUrl(ctx.project, loc, "google", entry.id, "predict"), {instances: [instance], parameters: {sample_count: 1}}, req.signal),
      {signal: req.signal},
    )
  } catch (err) {
    if (err.signIn) throw err
    return errorResponse("gemini", 502, `Lyria unreachable: ${err.message}`)
  }
  if (!res.ok) return failed(res)
  const [prediction] = (await res.json()).predictions ?? []
  // The docs name the field audioContent; bytesBase64Encoded is the Veo/Imagen spelling, accepted as a fallback.
  const audio = prediction?.audioContent ?? prediction?.bytesBase64Encoded
  if (!audio) return errorResponse("gemini", 502, "Lyria returned no audio")
  const mime = prediction.mimeType ?? "audio/wav"

  return geminiStream(async ({emit}) => {
    const saved = await saveMedia(ctx, {data: audio, mime, prefix: "lyria"})
    emit(
      chunk({
        text: `Generated a 30 s instrumental clip with ${entry.id}.\n[audio](${saved.url})\nPath: ${saved.path}`,
        inlineData: {mimeType: mime, data: audio},
        finishReason: "STOP",
        usage: USAGE,
      }),
    )
  }, {signal: req.signal})
}

// Upstream rejected the call before any stream opened: keep its status and Google's message.
async function failed(res) {
  const text = await res.text()
  let message = text
  try {
    message = JSON.parse(text).error?.message ?? text
  } catch {
    // Not JSON: the raw body is the message.
  }
  if (res.status === 400 && message.includes("Unsupported language detected")) message += ` ${LANGUAGE_HINT}`
  return errorResponse("gemini", res.status, message)
}

async function post(ctx, url, body, signal) {
  const token = await ctx.token()
  return ctx.fetch(url, {
    method: "POST",
    headers: {authorization: `Bearer ${token}`, "content-type": "application/json"},
    body: JSON.stringify(body),
    signal,
  })
}
