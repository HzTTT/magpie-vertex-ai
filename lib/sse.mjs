const encoder = new TextEncoder()

// Streams Gemini-protocol SSE. `data:` lines carry GenerateContentResponse objects. `: keepalive`
// comments only reach magpie (it counts them as liveness), so Gemini clients never see them.
export function geminiStream(run, {signal} = {}) {
  const aborter = new AbortController()
  let controller
  let done = false

  const end = () => {
    if (done) return
    done = true
    signal?.removeEventListener("abort", onAbort)
    controller.close()
  }
  const onAbort = () => {
    aborter.abort(signal.reason)
    end()
  }
  const write = (text) => {
    if (!done) controller.enqueue(encoder.encode(text))
  }
  const emit = (obj) => write(`data: ${JSON.stringify(obj)}\n\n`)

  const drive = async () => {
    try {
      await run({emit, keepalive: () => write(": keepalive\n\n"), signal: aborter.signal})
    } catch (err) {
      if (!aborter.signal.aborted) emit(chunk({text: `Error: ${err?.message ?? err}`, finishReason: "OTHER"}))
    } finally {
      end()
    }
  }

  if (signal?.aborted) aborter.abort(signal.reason)
  else signal?.addEventListener("abort", onAbort, {once: true})

  const body = new ReadableStream({
    start(c) {
      controller = c
      void drive()
    },
    cancel() {
      done = true
      signal?.removeEventListener("abort", onAbort)
      aborter.abort()
    },
  })
  return new Response(body, {headers: {"content-type": "text/event-stream"}})
}

// One GenerateContentResponse. `thought: true` marks progress text that Gemini clients show as reasoning.
export function chunk({text, thought = false, inlineData, finishReason, usage} = {}) {
  const parts = []
  if (text !== undefined) parts.push(thought ? {text, thought: true} : {text})
  if (inlineData) parts.push({inlineData})
  const candidate = {index: 0, content: {role: "model", parts}}
  if (finishReason) candidate.finishReason = finishReason
  const out = {candidates: [candidate]}
  if (usage) out.usageMetadata = usage
  return out
}
