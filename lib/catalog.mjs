// Vertex models this plugin routes, with the limits and modalities magpie needs. Discovery narrows the regions per project at runtime.
import { readFile } from "node:fs/promises"
import { join } from "node:path"
import { BASE } from "./regions.mjs"

const IN_ALL = ["text", "image", "audio", "video", "pdf"]
const IN_TI_PDF = ["text", "image", "pdf"]
const IN_TI_VP = ["text", "image", "video", "pdf"]
const IN_TI = ["text", "image"]
const IN_TEXT = ["text"]
const IN_AUDIO = ["audio"]
const IN_MM_EMBED = ["text", "image", "video"]

const OUT_TEXT = ["text"]
const OUT_TI = ["text", "image"]
const OUT_IMAGE = ["image"]
const OUT_AUDIO = ["audio"]
const OUT_VIDEO = ["video"]
const OUT_NONE = []

const DEFAULT_LIMITS = { context: 131072, output: 8192 }

const GEMINI_25_FLASH_REGIONS = ["global", "us-central1", "us-east5", "europe-west1", "europe-west4", "asia-northeast1", "asia-southeast1", "asia-northeast3", "asia-south1", "australia-southeast1", "europe-central2", "europe-north1", "europe-southwest1", "europe-west2", "europe-west3", "europe-west8", "europe-west9", "northamerica-northeast1", "southamerica-east1", "us-east1", "us-east4", "us-south1", "us-west1", "us-west4"]
const GEMINI_25_PRO_REGIONS = ["global", "us-central1", "us-east5", "europe-west1", "europe-west4"]
const TEXT_EMBEDDING_005_REGIONS = ["us-central1", "us-east5", "europe-west1", "europe-west4", "asia-northeast1", "asia-southeast1", "asia-east1", "asia-east2", "asia-northeast3", "asia-south1", "australia-southeast1", "europe-central2", "europe-north1", "europe-southwest1", "europe-west2", "europe-west3", "europe-west6", "europe-west8", "europe-west9", "me-central1", "me-central2", "me-west1", "northamerica-northeast1", "southamerica-east1", "us-east1", "us-east4", "us-south1", "us-west1", "us-west4"]
const MULTILINGUAL_EMBEDDING_REGIONS = ["us-central1", "us-east5", "europe-west1", "europe-west4", "asia-northeast1", "asia-southeast1"]
const MULTIMODAL_EMBEDDING_REGIONS = ["us-central1", "europe-west1", "europe-west4", "asia-northeast1", "asia-southeast1"]
const CLAUDE_3_OPUS_REGIONS = ["us-east5", "europe-west1", "europe-west4", "asia-southeast1"]

const GLOBAL = ["global"]
const GLOBAL_CENTRAL = ["global", "us-central1"]
const CENTRAL = ["us-central1"]

// Limits and capabilities come from Google's model docs; `// unverified` marks a value that is not in the docs and uses the default.
const RAW = [
  // unverified on Vertex: limits from the Gemini API page; reasoning and tool flags not documented, set to false
  { id: "gemini-2.5-computer-use-preview-10-2025", publisher: "google", family: "gemini", name: "Gemini 2.5 Computer Use Preview", regions: CENTRAL, context: 128000, output: 64000, reasoning: false, toolCall: false, input: IN_TI, outputs: OUT_TEXT, launchStage: "PUBLIC_PREVIEW" },
  { id: "gemini-2.5-flash", publisher: "google", family: "gemini", name: "Gemini 2.5 Flash", regions: GEMINI_25_FLASH_REGIONS, context: 1048576, output: 65536, reasoning: true, toolCall: true, input: IN_ALL, outputs: OUT_TEXT, launchStage: "GA" },
  { id: "gemini-2.5-flash-lite", publisher: "google", family: "gemini", name: "Gemini 2.5 Flash-Lite", regions: GEMINI_25_PRO_REGIONS, context: 1048576, output: 65536, reasoning: true, toolCall: true, input: IN_ALL, outputs: OUT_TEXT, launchStage: "GA" },
  { id: "gemini-2.5-pro", publisher: "google", family: "gemini", name: "Gemini 2.5 Pro", regions: GEMINI_25_PRO_REGIONS, context: 1048576, output: 65536, reasoning: true, toolCall: true, input: IN_ALL, outputs: OUT_TEXT, launchStage: "GA" },
  { id: "gemini-3-flash-preview", publisher: "google", family: "gemini", name: "Gemini 3 Flash Preview", regions: GLOBAL, context: 1048576, output: 65536, reasoning: true, toolCall: true, input: IN_ALL, outputs: OUT_TEXT, launchStage: "PUBLIC_PREVIEW" },
  { id: "gemini-3.1-flash-lite", publisher: "google", family: "gemini", name: "Gemini 3.1 Flash-Lite", regions: GLOBAL, context: 1048576, output: 65536, reasoning: true, toolCall: true, input: IN_ALL, outputs: OUT_TEXT, launchStage: "GA" },
  { id: "gemini-3.1-pro-preview", publisher: "google", family: "gemini", name: "Gemini 3.1 Pro Preview", regions: GLOBAL, context: 1048576, output: 65536, reasoning: true, toolCall: true, input: IN_ALL, outputs: OUT_TEXT, launchStage: "PUBLIC_PREVIEW" },
  { id: "gemini-3.5-flash", publisher: "google", family: "gemini", name: "Gemini 3.5 Flash", regions: GLOBAL, context: 1048576, output: 65536, reasoning: true, toolCall: true, input: IN_ALL, outputs: OUT_TEXT, launchStage: "GA" },
  { id: "gemini-3.5-flash-lite", publisher: "google", family: "gemini", name: "Gemini 3.5 Flash-Lite", regions: GLOBAL, context: 1048576, output: 65536, reasoning: true, toolCall: true, input: IN_ALL, outputs: OUT_TEXT, launchStage: "GA" },
  // unverified: no token limits published for the transcription model
  { id: "gemini-3.5-transcribe-preview", publisher: "google", family: "gemini", name: "Gemini 3.5 Transcribe Preview", regions: GLOBAL, context: 131072, output: 8192, reasoning: false, toolCall: false, input: IN_AUDIO, outputs: OUT_TEXT, launchStage: "PUBLIC_PREVIEW" },
  { id: "gemini-3.6-flash", publisher: "google", family: "gemini", name: "Gemini 3.6 Flash", regions: GLOBAL, context: 1048576, output: 65536, reasoning: true, toolCall: true, input: IN_ALL, outputs: OUT_TEXT, launchStage: "GA" },
  { id: "gemini-3.8-flash", publisher: "google", family: "gemini", name: "Gemini 3.8 Flash", regions: GLOBAL, context: 1048576, output: 65536, reasoning: true, toolCall: true, input: IN_ALL, outputs: OUT_TEXT, launchStage: "GA" },

  { id: "gemini-2.5-flash-image", publisher: "google", family: "gemini-image", name: "Gemini 2.5 Flash Image", regions: CENTRAL, context: 32768, output: 32768, reasoning: false, toolCall: false, input: IN_TI_PDF, outputs: OUT_TI, launchStage: "GA" },
  { id: "gemini-3-pro-image", publisher: "google", family: "gemini-image", name: "Gemini 3 Pro Image", regions: GLOBAL, context: 65536, output: 32768, reasoning: true, toolCall: false, input: IN_TI_PDF, outputs: OUT_TI, launchStage: "GA" },
  { id: "gemini-3.1-flash-image", publisher: "google", family: "gemini-image", name: "Gemini 3.1 Flash Image", regions: GLOBAL, context: 131072, output: 32768, reasoning: true, toolCall: false, input: IN_TI_VP, outputs: OUT_TI, launchStage: "GA" },
  { id: "gemini-3.1-flash-lite-image", publisher: "google", family: "gemini-image", name: "Gemini 3.1 Flash-Lite Image", regions: GLOBAL, context: 65536, output: 4096, reasoning: true, toolCall: false, input: IN_TI_VP, outputs: OUT_IMAGE, launchStage: "GA" },
  { id: "gemini-nano-banana-2.1", publisher: "google", family: "gemini-image", name: "Gemini Nano Banana 2.1", regions: GLOBAL, context: 131072, output: 32768, reasoning: true, toolCall: false, input: IN_TI_VP, outputs: OUT_TI, launchStage: "GA" },

  { id: "gemini-2.5-flash-tts", publisher: "google", family: "gemini-tts", name: "Gemini 2.5 Flash TTS", regions: GEMINI_25_PRO_REGIONS, context: 8192, output: 16384, reasoning: false, toolCall: false, input: IN_TEXT, outputs: OUT_AUDIO, launchStage: "GA" },
  { id: "gemini-2.5-pro-tts", publisher: "google", family: "gemini-tts", name: "Gemini 2.5 Pro TTS", regions: GEMINI_25_PRO_REGIONS, context: 8192, output: 16384, reasoning: false, toolCall: false, input: IN_TEXT, outputs: OUT_AUDIO, launchStage: "GA" },
  { id: "gemini-3.1-flash-tts-preview", publisher: "google", family: "gemini-tts", name: "Gemini 3.1 Flash TTS Preview", regions: CENTRAL, context: 8192, output: 16384, reasoning: false, toolCall: false, input: IN_TEXT, outputs: OUT_AUDIO, launchStage: "PUBLIC_PREVIEW" },

  // context is bytes per request, not tokens
  { id: "chirp3-hd", publisher: "google-cloud", family: "chirp-tts", name: "Chirp 3 HD TTS", regions: GLOBAL, context: 5000, output: 8192, reasoning: false, toolCall: false, input: IN_TEXT, outputs: OUT_AUDIO, launchStage: "GA" },

  { id: "gemini-embedding-001", publisher: "google", family: "embedding", name: "Gemini Embedding 001", regions: CENTRAL, context: 2048, output: 0, reasoning: false, toolCall: false, input: IN_TEXT, outputs: OUT_NONE, launchStage: "GA" },
  { id: "gemini-embedding-2", publisher: "google", family: "embedding", name: "Gemini Embedding 2", regions: GLOBAL, context: 8192, output: 0, reasoning: false, toolCall: false, input: IN_ALL, outputs: OUT_NONE, launchStage: "GA" },
  { id: "multimodalembedding", publisher: "google", family: "embedding", name: "Multimodal Embeddings", regions: MULTIMODAL_EMBEDDING_REGIONS, context: 32, output: 0, reasoning: false, toolCall: false, input: IN_MM_EMBED, outputs: OUT_NONE, launchStage: "GA" },
  { id: "text-embedding-005", publisher: "google", family: "embedding", name: "Text Embedding 005", regions: TEXT_EMBEDDING_005_REGIONS, context: 2048, output: 0, reasoning: false, toolCall: false, input: IN_TEXT, outputs: OUT_NONE, launchStage: "GA" },
  // unverified: absent from the Vertex embeddings table
  { id: "text-embedding-large-exp-03-07", publisher: "google", family: "embedding", name: "Text Embedding Large (exp 03-07)", regions: CENTRAL, context: 131072, output: 0, reasoning: false, toolCall: false, input: IN_TEXT, outputs: OUT_NONE, launchStage: "EXPERIMENTAL" },
  { id: "text-multilingual-embedding-002", publisher: "google", family: "embedding", name: "Text Multilingual Embedding 002", regions: MULTILINGUAL_EMBEDDING_REGIONS, context: 2048, output: 0, reasoning: false, toolCall: false, input: IN_TEXT, outputs: OUT_NONE, launchStage: "GA" },

  // unverified on Vertex: no prompt limit or output limit published, default used
  { id: "veo-2.0-generate-001", publisher: "google", family: "veo", name: "Veo 2.0", regions: ["us-central1", "europe-west3"], context: 131072, output: 8192, reasoning: false, toolCall: false, input: IN_TI, outputs: OUT_VIDEO, launchStage: "GA" },
  { id: "veo-3.0-generate-001", publisher: "google", family: "veo", name: "Veo 3.0", regions: CENTRAL, context: 131072, output: 8192, reasoning: false, toolCall: false, input: IN_TI, outputs: OUT_VIDEO, launchStage: "GA" },
  { id: "veo-3.0-fast-generate-001", publisher: "google", family: "veo", name: "Veo 3.0 Fast", regions: CENTRAL, context: 131072, output: 8192, reasoning: false, toolCall: false, input: IN_TI, outputs: OUT_VIDEO, launchStage: "GA" },
  // unverified on Vertex: 1024-token prompt limit is from the Gemini API page
  { id: "veo-3.1-generate-001", publisher: "google", family: "veo", name: "Veo 3.1", regions: CENTRAL, context: 1024, output: 8192, reasoning: false, toolCall: false, input: IN_TI, outputs: OUT_VIDEO, launchStage: "GA" },
  { id: "veo-3.1-fast-generate-001", publisher: "google", family: "veo", name: "Veo 3.1 Fast", regions: CENTRAL, context: 1024, output: 8192, reasoning: false, toolCall: false, input: IN_TI, outputs: OUT_VIDEO, launchStage: "GA" },
  { id: "veo-3.1-lite-generate-001", publisher: "google", family: "veo", name: "Veo 3.1 Lite", regions: CENTRAL, context: 1024, output: 8192, reasoning: false, toolCall: false, input: IN_TI, outputs: OUT_VIDEO, launchStage: "PUBLIC_PREVIEW" },

  // unverified: prompt limit not documented
  { id: "lyria-002", publisher: "google", family: "lyria", name: "Lyria 2", regions: CENTRAL, context: 131072, output: 8192, reasoning: false, toolCall: false, input: IN_TEXT, outputs: OUT_AUDIO, launchStage: "GA" },

  { id: "deepseek-ocr-maas", publisher: "deepseek-ai", family: "maas", name: "DeepSeek OCR (MaaS)", regions: GLOBAL, context: 8192, output: 8192, reasoning: false, toolCall: false, input: IN_TI, outputs: OUT_TEXT, launchStage: "GA" },
  // unverified: max output tokens not published on Vertex
  { id: "deepseek-r1-0528-maas", publisher: "deepseek-ai", family: "maas", name: "DeepSeek R1 0528 (MaaS)", regions: CENTRAL, context: 163840, output: 8192, reasoning: true, toolCall: true, input: IN_TEXT, outputs: OUT_TEXT, launchStage: "GA" },
  { id: "deepseek-v3.2-maas", publisher: "deepseek-ai", family: "maas", name: "DeepSeek V3.2 (MaaS)", regions: GLOBAL, context: 163840, output: 65536, reasoning: true, toolCall: true, input: IN_TEXT, outputs: OUT_TEXT, launchStage: "GA" },
  { id: "gemma-4-26b-a4b-it-maas", publisher: "google", family: "maas", name: "Gemma 4 26B A4B IT (MaaS)", regions: GLOBAL, context: 262144, output: 128000, reasoning: true, toolCall: true, input: IN_TI, outputs: OUT_TEXT, launchStage: "PUBLIC_PREVIEW" },
  { id: "minimax-m2-maas", publisher: "minimaxai", family: "maas", name: "MiniMax M2 (MaaS)", regions: GLOBAL, context: 196608, output: 196608, reasoning: true, toolCall: true, input: IN_TEXT, outputs: OUT_TEXT, launchStage: "GA" },
  { id: "kimi-k2-thinking-maas", publisher: "moonshotai", family: "maas", name: "Kimi K2 Thinking (MaaS)", regions: GLOBAL, context: 262144, output: 262144, reasoning: true, toolCall: true, input: IN_TEXT, outputs: OUT_TEXT, launchStage: "GA" },
  { id: "gpt-oss-120b-maas", publisher: "openai", family: "maas", name: "GPT-OSS 120B (MaaS)", regions: GLOBAL, context: 131072, output: 131072, reasoning: true, toolCall: true, input: IN_TEXT, outputs: OUT_TEXT, launchStage: "GA" },
  { id: "qwen3-235b-a22b-instruct-2507-maas", publisher: "qwen", family: "maas", name: "Qwen3 235B A22B Instruct 2507 (MaaS)", regions: GLOBAL, context: 262144, output: 16384, reasoning: false, toolCall: true, input: IN_TEXT, outputs: OUT_TEXT, launchStage: "GA" },
  { id: "qwen3-coder-480b-a35b-instruct-maas", publisher: "qwen", family: "maas", name: "Qwen3 Coder 480B A35B Instruct (MaaS)", regions: GLOBAL, context: 262144, output: 32768, reasoning: false, toolCall: true, input: IN_TEXT, outputs: OUT_TEXT, launchStage: "GA" },
  // unverified: max output tokens not published on Vertex
  { id: "qwen3-next-80b-a3b-instruct-maas", publisher: "qwen", family: "maas", name: "Qwen3 Next 80B A3B Instruct (MaaS)", regions: GLOBAL, context: 262144, output: 8192, reasoning: false, toolCall: true, input: IN_TEXT, outputs: OUT_TEXT, launchStage: "GA" },
  // unverified: max output tokens not published on Vertex
  { id: "qwen3-next-80b-a3b-thinking-maas", publisher: "qwen", family: "maas", name: "Qwen3 Next 80B A3B Thinking (MaaS)", regions: GLOBAL, context: 262144, output: 8192, reasoning: true, toolCall: true, input: IN_TEXT, outputs: OUT_TEXT, launchStage: "GA" },
  // unverified: max output tokens not published on Vertex
  { id: "grok-4.1-fast-non-reasoning", publisher: "xai", family: "maas", name: "Grok 4.1 Fast Non-Reasoning (MaaS)", regions: GLOBAL, context: 128000, output: 8192, reasoning: false, toolCall: true, input: IN_TI, outputs: OUT_TEXT, launchStage: "EXPERIMENTAL" },
  // unverified: max output tokens not published on Vertex
  { id: "grok-4.1-fast-reasoning", publisher: "xai", family: "maas", name: "Grok 4.1 Fast Reasoning (MaaS)", regions: GLOBAL, context: 128000, output: 8192, reasoning: true, toolCall: true, input: IN_TI, outputs: OUT_TEXT, launchStage: "EXPERIMENTAL" },
  // unverified: max output tokens not published on Vertex
  { id: "grok-4.20-non-reasoning", publisher: "xai", family: "maas", name: "Grok 4.20 Non-Reasoning (MaaS)", regions: GLOBAL, context: 2000000, output: 8192, reasoning: false, toolCall: true, input: IN_TI, outputs: OUT_TEXT, launchStage: "GA" },
  { id: "glm-4.7-maas", publisher: "zai-org", family: "maas", name: "GLM 4.7 (MaaS)", regions: GLOBAL, context: 200000, output: 128000, reasoning: false, toolCall: true, input: IN_TEXT, outputs: OUT_TEXT, launchStage: "GA" },
  { id: "glm-5-maas", publisher: "zai-org", family: "maas", name: "GLM 5 (MaaS)", regions: GLOBAL, context: 200000, output: 128000, reasoning: false, toolCall: true, input: IN_TEXT, outputs: OUT_TEXT, launchStage: "EXPERIMENTAL" },
  { id: "glm-5.2-maas", publisher: "zai-org", family: "maas", name: "GLM 5.2 (MaaS)", regions: GLOBAL, context: 1000000, output: 64000, reasoning: true, toolCall: true, input: IN_TEXT, outputs: OUT_TEXT, launchStage: "EXPERIMENTAL" },

  // unverified: Anthropic retired this version; limits and capabilities not in Vertex docs
  { id: "claude-3-opus@20240229", publisher: "anthropic", family: "claude", name: "Claude 3 Opus", regions: CLAUDE_3_OPUS_REGIONS, context: 131072, output: 8192, reasoning: false, toolCall: false, input: IN_TEXT, outputs: OUT_TEXT, launchStage: "GA" },
  { id: "claude-fable-5", publisher: "anthropic", family: "claude", name: "Claude Fable 5", regions: GLOBAL_CENTRAL, context: 1000000, output: 128000, reasoning: true, toolCall: true, input: IN_TI_PDF, outputs: OUT_TEXT, launchStage: "GA" },
  { id: "claude-fable-5-1", publisher: "anthropic", family: "claude", name: "Claude Fable 5.1", regions: GLOBAL_CENTRAL, context: 1000000, output: 128000, reasoning: true, toolCall: true, input: IN_TI_PDF, outputs: OUT_TEXT, launchStage: "GA" },
  { id: "claude-haiku-4-5@20251001", publisher: "anthropic", family: "claude", name: "Claude Haiku 4.5", regions: GLOBAL_CENTRAL, context: 200000, output: 64000, reasoning: true, toolCall: true, input: IN_TI_PDF, outputs: OUT_TEXT, launchStage: "GA" },
  { id: "claude-haiku-5-5", publisher: "anthropic", family: "claude", name: "Claude Haiku 5.5", regions: GLOBAL_CENTRAL, context: 1000000, output: 128000, reasoning: true, toolCall: true, input: IN_TI_PDF, outputs: OUT_TEXT, launchStage: "GA" },
  { id: "claude-opus-4-5@20251101", publisher: "anthropic", family: "claude", name: "Claude Opus 4.5", regions: GLOBAL_CENTRAL, context: 200000, output: 64000, reasoning: true, toolCall: true, input: IN_TI_PDF, outputs: OUT_TEXT, launchStage: "GA" },
  { id: "claude-opus-4-6", publisher: "anthropic", family: "claude", name: "Claude Opus 4.6", regions: GLOBAL_CENTRAL, context: 1000000, output: 128000, reasoning: true, toolCall: true, input: IN_TI_PDF, outputs: OUT_TEXT, launchStage: "GA" },
  { id: "claude-opus-4-7", publisher: "anthropic", family: "claude", name: "Claude Opus 4.7", regions: GLOBAL_CENTRAL, context: 1000000, output: 128000, reasoning: true, toolCall: true, input: IN_TI_PDF, outputs: OUT_TEXT, launchStage: "GA" },
  { id: "claude-opus-4-8", publisher: "anthropic", family: "claude", name: "Claude Opus 4.8", regions: GLOBAL_CENTRAL, context: 1000000, output: 128000, reasoning: true, toolCall: true, input: IN_TI_PDF, outputs: OUT_TEXT, launchStage: "GA" },
  { id: "claude-opus-5", publisher: "anthropic", family: "claude", name: "Claude Opus 5", regions: GLOBAL_CENTRAL, context: 1000000, output: 128000, reasoning: true, toolCall: true, input: IN_TI_PDF, outputs: OUT_TEXT, launchStage: "GA" },
  { id: "claude-opus-5-5", publisher: "anthropic", family: "claude", name: "Claude Opus 5.5", regions: GLOBAL_CENTRAL, context: 1000000, output: 128000, reasoning: true, toolCall: true, input: IN_TI_PDF, outputs: OUT_TEXT, launchStage: "GA" },
  { id: "claude-sonnet-4-6", publisher: "anthropic", family: "claude", name: "Claude Sonnet 4.6", regions: GLOBAL_CENTRAL, context: 1000000, output: 128000, reasoning: true, toolCall: true, input: IN_TI_PDF, outputs: OUT_TEXT, launchStage: "GA" },
  { id: "claude-sonnet-5", publisher: "anthropic", family: "claude", name: "Claude Sonnet 5", regions: GLOBAL_CENTRAL, context: 1000000, output: 128000, reasoning: true, toolCall: true, input: IN_TI_PDF, outputs: OUT_TEXT, launchStage: "GA" },
  { id: "claude-sonnet-5-5", publisher: "anthropic", family: "claude", name: "Claude Sonnet 5.5", regions: GLOBAL_CENTRAL, context: 1000000, output: 128000, reasoning: true, toolCall: true, input: IN_TI_PDF, outputs: OUT_TEXT, launchStage: "GA" },
]

// Defaults for a discovered id that the catalog does not know yet.
const FAMILY_DEFAULTS = {
  gemini: { ...DEFAULT_LIMITS, input: IN_ALL, outputs: OUT_TEXT, reasoning: false, toolCall: true },
  "gemini-image": { ...DEFAULT_LIMITS, input: IN_TI, outputs: OUT_TI, reasoning: false, toolCall: false },
  "gemini-tts": { ...DEFAULT_LIMITS, input: IN_TEXT, outputs: OUT_AUDIO, reasoning: false, toolCall: false },
  "chirp-tts": { ...DEFAULT_LIMITS, input: IN_TEXT, outputs: OUT_AUDIO, reasoning: false, toolCall: false },
  veo: { ...DEFAULT_LIMITS, input: IN_TI, outputs: OUT_VIDEO, reasoning: false, toolCall: false },
  lyria: { ...DEFAULT_LIMITS, input: IN_TEXT, outputs: OUT_AUDIO, reasoning: false, toolCall: false },
  embedding: { ...DEFAULT_LIMITS, output: 0, input: IN_TEXT, outputs: OUT_NONE, reasoning: false, toolCall: false },
  maas: { ...DEFAULT_LIMITS, input: IN_TEXT, outputs: OUT_TEXT, reasoning: false, toolCall: true },
  claude: { ...DEFAULT_LIMITS, input: IN_TI_PDF, outputs: OUT_TEXT, reasoning: false, toolCall: true },
}

const NPM_BY_FAMILY = { claude: "@ai-sdk/anthropic", maas: "@ai-sdk/openai-compatible", embedding: "@ai-sdk/openai-compatible" }
const BETA_STAGES = new Set(["PUBLIC_PREVIEW", "EXPERIMENTAL"])

const keyOf = (id) => id.split("@")[0]

export const CATALOG = RAW.map((e) => ({ ...e, key: keyOf(e.id), regions: orderRegions(e.regions) }))

export function npmFor(family) {
  return NPM_BY_FAMILY[family] ?? "@ai-sdk/google"
}

export function apiIdOf(entry) {
  return entry.family === "maas" ? `${entry.publisher}/${entry.id}` : entry.id
}

// Accepts a magpie key, a Vertex api id, or either with `@region`. Returns null for unknown ids and regions the entry lacks.
export function lookup(apiIdOrKey, list = CATALOG) {
  if (typeof apiIdOrKey !== "string") return null
  const exact = list.find((e) => apiIdOf(e) === apiIdOrKey || e.key === apiIdOrKey)
  if (exact) return { entry: exact, pinned: null }
  const at = apiIdOrKey.lastIndexOf("@")
  if (at <= 0) return null
  const base = apiIdOrKey.slice(0, at)
  const region = apiIdOrKey.slice(at + 1)
  const entry = list.find((e) => apiIdOf(e) === base || e.key === base)
  if (!entry || !entry.regions.includes(region)) return null
  return { entry, pinned: region }
}

export function configModels() {
  return Object.fromEntries(CATALOG.map((e) => [e.key, configModel(e)]))
}

// Model for the provider.models hook; `pinned` is a region the request must use.
export function toModel(entry, pinned = null) {
  const input = flags(entry.input, ["text", "image", "audio", "video", "pdf"])
  const output = flags(entry.outputs, ["text", "image", "audio", "video"])
  return {
    id: pinned ? `${entry.key}@${pinned}` : entry.key,
    name: pinned ? `${entry.name} (${pinned})` : entry.name,
    api: { id: apiIdOf(entry), url: BASE, npm: npmFor(entry.family) },
    limit: { context: entry.context, output: entry.output },
    capabilities: {
      reasoning: entry.reasoning,
      toolcall: entry.toolCall,
      attachment: input.image || input.audio || input.video || input.pdf,
      input,
      output,
    },
    cost: { input: 0, output: 0, cache: { read: 0, write: 0 } },
    status: BETA_STAGES.has(entry.launchStage) ? "beta" : "active",
    ...(entry.reasoning && { variants: variants() }),
  }
}

// `include`/`exclude` take magpie keys or /regex/ strings. With regionVariants, every region of an entry also gets a `key@region` model.
// Embedding models are listed last: magpie's `provider test` probes the first listed model that speaks a chat API.
export function modelsFor({ entries = CATALOG, options = {} } = {}) {
  const include = matchers(options.include)
  const exclude = matchers(options.exclude)
  const models = {}
  const ordered = [...entries.filter((e) => e.family !== "embedding"), ...entries.filter((e) => e.family === "embedding")]
  for (const entry of ordered) {
    if (include && !include.some((match) => match(entry.key))) continue
    if (exclude?.some((match) => match(entry.key))) continue
    models[entry.key] = toModel(entry)
    if (options.regionVariants) {
      for (const loc of entry.regions) models[`${entry.key}@${loc}`] = toModel(entry, loc)
    }
  }
  return models
}

export function discoveredFile(dir, project) {
  return join(dir, "vertex-ai", `catalog-${project}.json`)
}

// Null when the file is missing or unreadable; the caller then uses the static catalog.
export async function loadDiscovered(dir, project) {
  try {
    return JSON.parse(await readFile(discoveredFile(dir, project), "utf8"))
  } catch {
    return null
  }
}

export function mergeDiscovered(discovered) {
  if (!discovered) return CATALOG
  const byKey = new Map()
  for (const d of discovered.models ?? []) {
    const entry = discoveredEntry(d)
    if (!entry) continue
    const prev = byKey.get(entry.key)
    // Two versions of one model share a key; the one the catalog names wins.
    if (!prev || (!fromCatalog(prev) && fromCatalog(entry))) byKey.set(entry.key, entry)
  }
  return [...byKey.values()]
}

// Puts global first, then us-central1, then the rest in their given order.
export function orderRegions(regions) {
  const rank = (loc) => (loc === "global" ? 0 : loc === "us-central1" ? 1 : 2)
  return [...regions].sort((a, b) => rank(a) - rank(b))
}

function discoveredEntry(d) {
  const regions = orderRegions(d.regions ?? [])
  const known = CATALOG.find((e) => e.id === d.id && e.publisher === d.publisher)
  if (known) return { ...known, regions }
  const defaults = FAMILY_DEFAULTS[d.family]
  if (!defaults) return null
  return { ...defaults, id: d.id, key: keyOf(d.id), publisher: d.publisher, family: d.family, name: d.id, regions }
}

function fromCatalog(entry) {
  return CATALOG.some((e) => e.id === entry.id && e.publisher === entry.publisher)
}

function configModel(entry) {
  return {
    name: entry.name,
    id: apiIdOf(entry),
    limit: { context: entry.context, output: entry.output },
    reasoning: entry.reasoning,
    tool_call: entry.toolCall,
    modalities: { input: entry.input, output: entry.outputs },
    provider: { npm: npmFor(entry.family), api: BASE },
    ...(entry.reasoning && { variants: variants() }),
  }
}

function variants() {
  return { low: {}, medium: {}, high: {} }
}

function flags(list, names) {
  return Object.fromEntries(names.map((name) => [name, list.includes(name)]))
}

function matchers(list) {
  if (!list?.length) return null
  return list.map((pattern) => {
    if (/^\/.+\/$/.test(pattern)) {
      const re = new RegExp(pattern.slice(1, -1))
      return (key) => re.test(key)
    }
    return (key) => key === pattern
  })
}
