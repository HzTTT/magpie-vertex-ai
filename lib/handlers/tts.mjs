import {modelUrl} from "../regions.mjs"
import {regionsFor, withRegions} from "../failover.mjs"
import {errorResponse} from "../errors.mjs"
import {parsePcmMime, pcmToWav, saveMedia} from "../media.mjs"
import {lastUserTurn} from "../prompt.mjs"
import {chunk, geminiStream} from "../sse.mjs"

const USAGE = {promptTokenCount: 0, candidatesTokenCount: 0, totalTokenCount: 0}
const CJK = /[\u3400-\u9fff\uf900-\ufaff]/

export async function handle(req, ctx) {
  const {text} = lastUserTurn(req.body)
  if (!text.trim()) return errorResponse("gemini", 400, "Text-to-speech needs text in the last user turn")
  const {entry, pinned} = req
  // Synthesis happens before the stream opens, so an upstream failure is an HTTP error magpie can fail over on.
  const speech =
    entry.family === "chirp-tts"
      ? await chirp(ctx, text, req.signal)
      : await geminiTts(ctx, entry, pinned, text, req.signal)
  if (speech instanceof Response) return speech
  return geminiStream(async ({emit}) => {
    const saved = await saveMedia(ctx, {data: speech.bytes, mime: speech.mime, prefix: "tts"})
    emit(
      chunk({
        text: `${speech.label}\n[audio](${saved.url})\nPath: ${saved.path}`,
        inlineData: {mimeType: speech.mime, data: Buffer.from(speech.bytes).toString("base64")},
        finishReason: "STOP",
        usage: USAGE,
      }),
    )
  }, {signal: req.signal})
}

// Returns {bytes, mime, label} on success, or an HTTP error Response.
async function geminiTts(ctx, entry, pinned, text, signal) {
  const voice = ctx.options.ttsVoice ?? "Kore"
  const body = {
    contents: [{role: "user", parts: [{text}]}],
    generationConfig: {
      responseModalities: ["AUDIO"],
      speechConfig: {voiceConfig: {prebuiltVoiceConfig: {voiceName: voice}}},
    },
  }
  let res
  try {
    res = await withRegions(
      regionsFor(entry, ctx, pinned),
      (loc) => post(ctx, modelUrl(ctx.project, loc, "google", entry.id, "generateContent"), body, signal),
      {signal},
    )
  } catch (err) {
    if (err.signIn) throw err
    return errorResponse("gemini", 502, `Gemini TTS unreachable: ${err.message}`)
  }
  if (!res.ok) return failed(res)
  const inline = (await res.json()).candidates?.[0]?.content?.parts?.find((p) => p.inlineData?.data)?.inlineData
  if (!inline) return errorResponse("gemini", 502, "Gemini TTS returned no audio")
  const label = `Speech generated with ${entry.id} (voice ${voice}).`
  const pcm = parsePcmMime(inline.mimeType)
  if (pcm) return {bytes: pcmToWav(Buffer.from(inline.data, "base64"), pcm), mime: "audio/wav", label}
  if (/^audio\/(x-)?wav$/i.test(inline.mimeType)) return {bytes: Buffer.from(inline.data, "base64"), mime: "audio/wav", label}
  return errorResponse("gemini", 502, `Unexpected Gemini TTS audio type: ${inline.mimeType}`)
}

// Cloud Text-to-Speech is a global API, so no region failover here.
async function chirp(ctx, text, signal) {
  const language = ctx.options.chirpLanguage ?? (CJK.test(text) ? "cmn-CN" : "en-US")
  const voice = `${language}-Chirp3-HD-${ctx.options.chirpVoice ?? "Achernar"}`
  let res
  try {
    res = await ctx.fetch("https://texttospeech.googleapis.com/v1/text:synthesize", {
      method: "POST",
      headers: {
        authorization: `Bearer ${await ctx.token()}`,
        "content-type": "application/json",
        "x-goog-user-project": ctx.project,
      },
      body: JSON.stringify({
        input: {text},
        voice: {languageCode: language, name: voice},
        audioConfig: {audioEncoding: "MP3"},
      }),
      signal,
    })
  } catch (err) {
    if (err.signIn) throw err
    return errorResponse("gemini", 502, `Chirp TTS unreachable: ${err.message}`)
  }
  if (!res.ok) return failed(res)
  const {audioContent} = await res.json()
  if (!audioContent) return errorResponse("gemini", 502, "Chirp TTS returned no audio")
  return {bytes: Buffer.from(audioContent, "base64"), mime: "audio/mpeg", label: `Speech generated with Chirp 3 HD (${voice}).`}
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
