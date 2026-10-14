import { regionsFor, withRegions } from "../failover.mjs"
import { lookup } from "../catalog.mjs"
import { modelUrl } from "../regions.mjs"
import { errorResponse } from "../errors.mjs"

// Inline ceiling for generateContent is 20 MB per request; base64 grows the file
// by 4/3, so the raw audio must stay under 15 MB. Larger files need the Files API.
const MAX_BYTES = 15 * 1024 * 1024
const FORMATS = new Set(["json", "text", "srt", "vtt", "verbose_json"])
const TIMED_MODEL = "gemini-2.5-flash"
const MIME_ALIAS = {
  "audio/mpeg": "audio/mp3",
  "audio/x-wav": "audio/wav",
  "audio/wave": "audio/wav",
  "audio/m4a": "audio/mp4",
}
const MIME_BY_EXT = {
  mp3: "audio/mp3",
  wav: "audio/wav",
  aiff: "audio/aiff",
  aif: "audio/aiff",
  aac: "audio/aac",
  ogg: "audio/ogg",
  opus: "audio/ogg",
  flac: "audio/flac",
  m4a: "audio/mp4",
  mp4: "video/mp4",
  mov: "video/mov",
  webm: "video/webm",
}

export async function handle(req, ctx) {
  const form = req.form
  if (!form || typeof form.get !== "function") {
    return errorResponse("openai", 400, "multipart/form-data body is required")
  }
  const file = form.get("file")
  if (!(file instanceof Blob)) return errorResponse("openai", 400, "file is required")
  if (file.size > MAX_BYTES) {
    return errorResponse("openai", 413, `file is ${file.size} bytes; inline audio is limited to 15 MB`)
  }

  const format = textField(form, "response_format") ?? "json"
  if (!FORMATS.has(format)) return errorResponse("openai", 400, `unsupported response_format: ${format}`)
  const rawTemperature = textField(form, "temperature")
  const temperature = rawTemperature === undefined ? undefined : Number(rawTemperature)
  if (temperature !== undefined && !(Number.isFinite(temperature) && temperature >= 0)) {
    return errorResponse("openai", 400, "temperature must be a non-negative number")
  }
  const mimeType = mimeOf(file)
  if (!mimeType) {
    return errorResponse("openai", 400, "cannot determine audio type; set the file Content-Type or a known extension")
  }

  const translate = req.suffix.startsWith("/audio/translations")
  const segmented = format === "srt" || format === "vtt" || format === "verbose_json"
  const language = textField(form, "language")
  const body = {
    contents: [{
      role: "user",
      parts: [
        { inlineData: { mimeType, data: Buffer.from(await file.arrayBuffer()).toString("base64") } },
        { text: instruction({ translate, segmented, language, prompt: textField(form, "prompt") }) },
      ],
    }],
    generationConfig: { temperature },
  }

  // gemini-3.5-transcribe-preview answers plain text only, so timed formats run on a general Gemini model.
  const entry = segmented && req.entry.id.includes("transcribe") ? lookup(TIMED_MODEL).entry : req.entry
  const regions = regionsFor(entry, ctx, entry === req.entry ? req.pinned : undefined)
  const token = await ctx.token()
  const res = await withRegions(regions, loc => ctx.fetch(
    modelUrl(ctx.project, loc, entry.publisher, entry.id, "generateContent"),
    {
      method: "POST",
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify(body),
      signal: req.signal,
    },
  ), { signal: req.signal })
  if (!res.ok) return errorResponse("openai", res.status, upstreamMessage(await res.text()))

  const reply = replyText(await res.json())
  if (!segmented) return plainReply(format, reply)

  const parsed = parseJsonReply(reply)
  if (!Array.isArray(parsed?.segments)) return errorResponse("openai", 502, "malformed transcription response")
  const task = translate ? "translate" : "transcribe"
  return segmentReply(format, {
    task,
    language: translate ? "english" : (parsed.language ?? language ?? ""),
    segments: parsed.segments.map((s, id) => ({
      id,
      start: Number(s.start),
      end: Number(s.end),
      text: String(s.text ?? "").trim(),
    })),
  })
}

function instruction({ translate, segmented, language, prompt }) {
  const lines = [translate
    ? "Translate the speech in this audio into English."
    : "Transcribe the speech in this audio verbatim, in its original language."]
  if (language) lines.push(`The speech is in language: ${language}.`)
  if (prompt) lines.push(`Context to help with names and terms: ${prompt}`)
  lines.push(segmented
    ? 'Return only JSON: {"language": <source language code>, "segments": [{"start": <seconds>, "end": <seconds>, "text": <text>}]}, with times measured from the start of the audio.'
    : `Output only the ${translate ? "English translation" : "transcript"}.`)
  return lines.join("\n")
}

function plainReply(format, text) {
  if (format === "json") return Response.json({ text })
  return new Response(text, { headers: { "content-type": "text/plain; charset=utf-8" } })
}

function segmentReply(format, { task, language, segments }) {
  if (format === "srt") {
    return new Response(srt(segments), { headers: { "content-type": "text/plain; charset=utf-8" } })
  }
  if (format === "vtt") {
    return new Response(vtt(segments), { headers: { "content-type": "text/vtt; charset=utf-8" } })
  }
  return Response.json({
    task,
    language,
    duration: segments.at(-1)?.end ?? 0,
    text: segments.map(s => s.text).join(" "),
    segments,
  })
}

function srt(segments) {
  return segments
    .map((s, i) => `${i + 1}\n${stamp(s.start, ",")} --> ${stamp(s.end, ",")}\n${s.text}\n`)
    .join("\n")
}

function vtt(segments) {
  return "WEBVTT\n\n" + segments
    .map(s => `${stamp(s.start, ".")} --> ${stamp(s.end, ".")}\n${s.text}\n`)
    .join("\n")
}

// HH:MM:SS<sep>mmm, with sep "," for SRT and "." for WebVTT.
function stamp(seconds, sep) {
  const ms = Math.round(seconds * 1000)
  const pad = (n, width = 2) => String(n).padStart(width, "0")
  return `${pad(Math.floor(ms / 3600000))}:${pad(Math.floor(ms / 60000) % 60)}:${pad(Math.floor(ms / 1000) % 60)}${sep}${pad(ms % 1000, 3)}`
}

function mimeOf(file) {
  const declared = file.type && file.type !== "application/octet-stream" ? file.type : undefined
  if (declared) return MIME_ALIAS[declared] ?? declared
  const ext = /\.([a-z0-9]+)$/i.exec(file.name ?? "")?.[1].toLowerCase()
  return MIME_BY_EXT[ext]
}

function textField(form, name) {
  const value = form.get(name)
  return typeof value === "string" && value !== "" ? value : undefined
}

// Concatenates the answer text, skipping thought parts.
function replyText(json) {
  const parts = json?.candidates?.[0]?.content?.parts ?? []
  return parts.filter(p => !p.thought).map(p => p.text ?? "").join("").trim()
}

// Models that reject a JSON response schema for audio may still fence the answer in a code block.
function parseJsonReply(text) {
  const body = text.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "")
  try {
    return JSON.parse(body)
  } catch {
    return null
  }
}

function upstreamMessage(text) {
  try {
    const message = JSON.parse(text)?.error?.message
    if (message) return message
  } catch {
    // not JSON; fall through to the raw text
  }
  return text.slice(0, 500) || "upstream request failed"
}
