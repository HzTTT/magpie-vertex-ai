// Vertex AI serving locations, from the probe of this project (vertex-inventory.md).
// us, eu, ca and asia-southeast3 answer 404 for every publisher, so they are not listed.
export const BASE = "https://vertex-ai.magpie.invalid/v1"

// global first (most models serve there), then us-central1, then the rest.
export const REGIONS = [
  "global",
  "us-central1",
  "africa-south1",
  "asia-east1",
  "asia-east2",
  "asia-northeast1",
  "asia-northeast2",
  "asia-northeast3",
  "asia-south1",
  "asia-south2",
  "asia-southeast1",
  "asia-southeast2",
  "australia-southeast1",
  "australia-southeast2",
  "europe-central2",
  "europe-north1",
  "europe-north2",
  "europe-southwest1",
  "europe-west1",
  "europe-west10",
  "europe-west12",
  "europe-west15",
  "europe-west2",
  "europe-west3",
  "europe-west4",
  "europe-west6",
  "europe-west8",
  "europe-west9",
  "me-central1",
  "me-central2",
  "me-west1",
  "northamerica-northeast1",
  "northamerica-northeast2",
  "southamerica-east1",
  "southamerica-west1",
  "us-central2",
  "us-east1",
  "us-east4",
  "us-east5",
  "us-east7",
  "us-south1",
  "us-west1",
  "us-west2",
  "us-west3",
  "us-west4",
  "us-west8",
]

export function hostFor(loc) {
  return loc === "global" ? "aiplatform.googleapis.com" : `${loc}-aiplatform.googleapis.com`
}

export function locationBase(project, loc, version = "v1") {
  return `https://${hostFor(loc)}/${version}/projects/${project}/locations/${loc}`
}

export function modelUrl(project, loc, publisher, model, method, version = "v1") {
  return `${locationBase(project, loc, version)}/publishers/${publisher}/models/${model}:${method}`
}

export function openapiUrl(project, loc, path) {
  return `${locationBase(project, loc, "v1beta1")}/endpoints/openapi${path}`
}
