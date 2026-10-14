// Newest role:"user" content of a Gemini GenerateContentRequest. Media is bucketed by mime major type and
// sorted by mime within each bucket.
export function lastUserTurn(geminiBody) {
  const contents = Array.isArray(geminiBody?.contents) ? geminiBody.contents : []
  const turn = contents.findLast((c) => c?.role === "user")
  const parts = Array.isArray(turn?.parts) ? turn.parts : []
  const texts = []
  const media = {image: [], audio: [], video: []}
  for (const part of parts) {
    if (typeof part?.text === "string") texts.push(part.text)
    const inline = part?.inlineData
    if (!inline?.data) continue
    const mime = inline.mimeType ?? ""
    const kind = mime.split("/")[0]
    if (Object.hasOwn(media, kind)) media[kind].push({mime, data: inline.data})
  }
  const bySortedMime = (a, b) => (a.mime < b.mime ? -1 : a.mime > b.mime ? 1 : 0)
  return {
    text: texts.join("\n"),
    images: media.image.sort(bySortedMime),
    audio: media.audio.sort(bySortedMime),
    video: media.video.sort(bySortedMime),
  }
}
