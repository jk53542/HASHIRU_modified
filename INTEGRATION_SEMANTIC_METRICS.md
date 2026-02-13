# Integrating Semantic Density and Semantic Entropy with HASHIRU

HASHIRU uses **semantic entropy** (from the semantic_uncertainty repo) and **semantic density** (from the semantic-density-paper repo) as metrics. All three projects depend on different versions of the same libraries (e.g. `torch`, `transformers`, `numpy`), so they cannot be imported in a single process. The recommended approach is to run the two semantic repos as **separate HTTP services** in their own environments and have HASHIRU call them from `semantic_metrics.py`.

**Do the semantic repos need the weights of the model that generated the responses?** No. They only need the **response text**. Semantic entropy uses an **entailment model** (e.g. DeBERTa MNLI) inside the semantic_uncertainty repo to compare and cluster responses. Semantic density uses **sentence embeddings** (e.g. SentenceTransformer) inside the semantic-density repo to compute similarity between responses. Neither uses your agent’s generative model (Ollama, Groq, etc.), so you do not pass agent model weights to these services.

---

## 1. What inputs do the two repos need?

### Semantic uncertainty (entropy)

- **Location:** `semantic_uncertainty/semantic_uncertainty/`
- **Core functions:**  
  - `get_semantic_ids(responses, model=entailment_model, ...)`  
  - `cluster_assignment_entropy(semantic_ids)`  
  - Optional: `logsumexp_by_id(semantic_ids, log_likelihoods, ...)` → `predictive_entropy_rao(...)` for full Rao entropy

**Inputs:**

| Input | Required | Description |
|-------|----------|-------------|
| **responses** | Yes | List of strings: multiple answer samples for the same question (e.g. `[response, sample1, sample2, ...]`). |
| **prompt / question** | Yes (for LLM entailment) | Used as context for entailment (e.g. DeBERTa can work without it; GPT/LLaMA entailment uses it). |
| **log_likelihoods** | No (for cluster-only entropy) | Per-response token log-likelihoods. Needed only for full *predictive* semantic entropy (Rao). Without them you can still compute cluster-assignment entropy. |
| **entailment model** | Yes | Loaded inside that repo (e.g. `EntailmentDeberta()`, or GPT-4 / LLaMA). |

So for a minimal integration you send: **prompt** (question) + **responses** (list of strings). The service running in the semantic_uncertainty env will load the entailment model, compute `get_semantic_ids(responses, ...)`, then `cluster_assignment_entropy(semantic_ids)` (and optionally Rao entropy if you also send log likelihoods).

### Semantic density

- **Location:** `semantic-density-paper/experiment_code/`
- **Core logic:**  
  - `semantic_metrics.py`: `compute_semantic_density_from_similarity_matrix(S)` (density = mean off-diagonal similarity).  
  - `compute_semantic_density_from_text_responses(responses)` builds embeddings (or uses provided ones), then similarity matrix, then density.  
  - The full paper pipeline uses DeBERTa NLI (question+response encoding) and likelihood-weighted averaging.

**Inputs:**

| Input | Required | Description |
|-------|----------|-------------|
| **responses** | Yes | List of strings: the set of alternative responses (e.g. `[response, sample1, sample2, ...]`). |
| **prompt / question** | Optional | Used in the full paper pipeline (question + each response concatenated for NLI). For the simpler “similarity matrix” path, only responses (or embeddings) are needed. |

So for a minimal integration you send: **prompt** + **responses**. The service in the semantic-density env can implement either the simple similarity-based density or the full NLI-based pipeline.

---

## 2. How HASHIRU feeds these inputs

In `semantic_metrics.py`, HASHIRU already builds a single payload and calls one service URL:

- **prompt:** user/context prompt (string).  
- **response:** the main agent response (string).  
- **samples:** optional list of extra response strings (e.g. multiple samples for the same prompt).

The payload sent to the service is:

```json
{
  "prompt": "<prompt>",
  "responses": ["<response>"],
  "samples": [["<sample1>", "<sample2>", ...]],
  "metadata": {}
}
```

So **responses** = `[response]` and **samples[i]** = list of alternatives for the i-th response. The service interprets this as:

- **For entropy:** use `responses[0]` plus `samples[0]` (if present) as the full list of answer strings for that prompt, then compute semantic entropy over that set.  
- **For density:** same set of strings to compute semantic density (e.g. similarity matrix over `responses[0]` + `samples[0]`).

**Important:** Both metrics are defined over *multiple* responses for the same prompt (see [semantic entropy](https://arxiv.org/html/2405.19648v1) and [semantic density](https://arxiv.org/html/2405.13845v3)). If you send only one response (no samples), you will get **entropy = 0.0** and **density = 1.0** by convention; to get meaningful, varying values you must pass **multiple samples** (alternative responses for the same prompt), e.g. by sampling your model several times for that prompt and passing the extra outputs as `samples`.

So you are **feeding the two repos** by:

1. Running a metrics service that receives this JSON (either the existing `service_wrapper` or one service per repo).  
2. Inside that service (in the appropriate env), building the list of strings: `all_responses = [response] + (samples[0] or [])`.  
3. Calling the semantic_uncertainty code with `all_responses` (and prompt) for entropy, and the semantic_density code with `all_responses` (and optionally prompt) for density.  
4. Returning `entropy` and `density` in the response (e.g. one value each, or lists of length 1).

---

## 3. Dealing with different library versions: run separate processes

Because the three projects use different versions of the same libraries:

- **Do not** try to install all three in one environment or import semantic_uncertainty / semantic_density code directly inside HASHIRU.  
- **Do** run the semantic logic in separate processes, each in its own environment (venv/conda), and talk over HTTP (or another IPC).

Options:

### Option A: Single unified service (current `service_wrapper`)

- **Idea:** One FastAPI app (e.g. `service_wrapper/comm_app.py`) that implements `/score` and internally calls both entropy and density.  
- **Problem:** That single process would need both repos’ dependencies, so you still get version conflicts unless you isolate the heavy work (e.g. subprocess or second service).

So in practice you either:

- Keep the unified service as a **thin gateway** that forwards to two backend services (see Option B), or  
- Implement **Option B** and have HASHIRU call two URLs (one for entropy, one for density) if you prefer not to run a gateway.

### Option B: Two services (recommended)

- **Service 1 – Semantic entropy**  
  - Environment: use `semantic_uncertainty`’s env (e.g. `environment_export.yaml`).  
  - Small FastAPI app that:  
    - Loads the entailment model once.  
    - Exposes e.g. `POST /score` or `POST /entropy` with `{ "prompt", "responses", "samples" }`.  
    - Builds `all_responses = responses[0] + (samples[0] or [])`, calls `get_semantic_ids(all_responses, ...)`, then `cluster_assignment_entropy(semantic_ids)` (and optionally Rao entropy if you pass log_likelihoods).  
    - Returns `{ "entropy": <float>, "clusters": ... }`.

- **Service 2 – Semantic density**  
  - Environment: use `semantic-density-paper`’s env (e.g. `environment_llama3.yml` or the one you use for that repo).  
  - Small FastAPI app that:  
    - Loads any needed model (e.g. for embeddings/NLI) once.  
    - Exposes e.g. `POST /score` or `POST /density` with `{ "prompt", "responses", "samples" }`.  
    - Builds the same `all_responses`, calls `compute_semantic_density_from_text_responses(all_responses)` (or the full NLI pipeline).  
    - Returns `{ "density": <float>, "kernel_values": ... }`.

Then either:

- **Gateway:** Keep `service_wrapper/comm_app.py` in HASHIRU’s env; it receives one `/score` from HASHIRU, calls Service 1 and Service 2, and merges results into `{ "entropy": [...], "density": [...] }`. No heavy deps in the gateway.  
- **Two URLs from HASHIRU:** In `semantic_metrics.py`, call the entropy service for `compute_semantic_entropy` and the density service for `compute_semantic_density` (each with the same payload). You can set `HASHIRU_METRICS_SERVICE_URL` to the gateway, or use two env vars (e.g. `HASHIRU_ENTROPY_SERVICE_URL` and `HASHIRU_DENSITY_SERVICE_URL`) if you split the endpoints.

---

## Option B: Concrete steps to run two services + gateway

Follow these steps to get Option B running end-to-end.

### 1. Install FastAPI and uvicorn in both backend envs

- **Entropy env:**  
  `conda activate semantic_uncertainty`  
  `pip install fastapi uvicorn`

- **Density env:**  
  `conda activate semantic_density`  
  `pip install fastapi uvicorn`

The gateway runs in **HASHIRU’s env** (which already has FastAPI/uvicorn and `requests`).

### 2. Service 1 – Semantic entropy (port 8124)

- **Where:** `semantic_uncertainty/semantic_uncertainty/entropy_service.py` (in the semantic_uncertainty repo).
- **Environment:** `semantic_uncertainty` conda env.
- **Start:**

  ```bash
  conda activate semantic_uncertainty
  cd /path/to/hashiru_modified/semantic_uncertainty/semantic_uncertainty
  export PYTHONPATH="$(pwd)"
  uvicorn entropy_service:app --host 127.0.0.1 --port 8124
  ```

- **Check:** `curl http://127.0.0.1:8124/health` → `{"status":"ok","service":"entropy"}`.  
  The first request will be slow while the DeBERTa model loads.

### 3. Service 2 – Semantic density (port 8125)

- **Where:** `semantic-density-paper/experiment_code/density_service.py`.
- **Environment:** `semantic_density` conda env.
- **Start:**

  ```bash
  conda activate semantic_density
  cd /path/to/hashiru_modified/semantic-density-paper/experiment_code
  uvicorn density_service:app --host 127.0.0.1 --port 8125
  ```

- **Check:** `curl http://127.0.0.1:8125/health` → `{"status":"ok","service":"density"}`.

### 4. Gateway (port 8123) – run in HASHIRU’s env

- **Where:** `service_wrapper/comm_app.py`.
- **Environment:** Your HASHIRU virtualenv (same as when you run the HASHIRU app).
- **Start:** From the **parent directory of** `service_wrapper` (e.g. the repo root that contains both `HASHIRU_modified` and `service_wrapper`):

  ```bash
  # Activate HASHIRU's venv, then:
  cd /path/to/hashiru_modified
  uvicorn service_wrapper.comm_app:app --host 127.0.0.1 --port 8123
  ```

- **Check:** `curl http://127.0.0.1:8123/health` → should report `entropy_up` and `density_up` (true when both backends are running).

### 5. Point HASHIRU at the gateway

- Set the base URL for the single metrics endpoint HASHIRU calls:

  ```bash
  export HASHIRU_METRICS_SERVICE_URL=http://127.0.0.1:8123
  ```

- Optional: `HASHIRU_METRICS_TIMEOUT=60` (or higher if DeBERTa is slow).  
  HASHIRU’s `semantic_metrics.py` already calls `POST {HASHIRU_METRICS_SERVICE_URL}/score` and expects `{ "entropy": [...], "density": [...] }`; the gateway provides that by calling the two backends.

### 6. Start order and quick test

1. Start **entropy** (8124), then **density** (8125), then **gateway** (8123).  
2. Start HASHIRU (with `HASHIRU_METRICS_SERVICE_URL=http://127.0.0.1:8123`).  
3. Trigger a flow that uses semantic metrics; the gateway will forward to both services and return merged entropy and density.

**Optional env vars for the gateway** (if backends run elsewhere):

- `ENTROPY_SERVICE_URL` (default `http://127.0.0.1:8124`)
- `DENSITY_SERVICE_URL` (default `http://127.0.0.1:8125`)
- `HASHIRU_METRICS_BACKEND_TIMEOUT` (default `120` seconds)

---

### Option C: Subprocess (no HTTP)

- From HASHIRU, run a script in the other env, e.g.  
  `subprocess.run(["path/to/other/venv/bin/python", "entropy_script.py"], input=json.dumps(payload), ...)`  
- The script reads JSON from stdin, imports the semantic_uncertainty (or density) code, computes the metric, prints JSON to stdout.  
- Works but is slower and more awkward than a long-lived HTTP service (model loaded once per process).

---

## Troubleshooting: constant 0.0 entropy and 0.5 / 1.0 density

If every example returns **entropy = 0.0** and **density = 0.5** (or **density = 1.0** after the single-response convention), there are two common causes:

1. **No samples provided**  
   Both metrics are defined over *multiple* responses for the same prompt. If you only send one response (no `samples`), the backends return entropy 0.0 and density 1.0 by convention. **Fix:** Pass multiple alternative responses for the same prompt in the `samples` parameter (e.g. generate 3–5 responses from your model for that prompt and pass them as a comma-separated list in the tool, or as `samples` in the API).

2. **Backend services not running or unreachable**  
   If the gateway cannot reach the entropy (8124) or density (8125) services, it still returns 200 with default values (0.0 entropy, 0.5 density). **Fix:** Start both backend services and the gateway (see Option B steps above). Check `curl http://127.0.0.1:8123/health` and ensure `entropy_up` and `density_up` are true. The tool output now includes `diagnostics.entropy_ok`, `diagnostics.density_ok`, and any `entropy_error` / `density_error` so you can see if a backend failed.

---

## 4. Summary

| Question | Answer |
|----------|--------|
| **What inputs do the two semantic repos need?** | **Entropy:** prompt + list of response strings (and optionally per-response log likelihoods). **Density:** list of response strings (and optionally prompt). |
| **How does HASHIRU feed those inputs?** | It sends one JSON body: `prompt`, `responses` (e.g. `[response]`), `samples` (e.g. `[[s1, s2, ...]]`). The service builds `all_responses = responses[0] + (samples[0] or [])` and passes that (and prompt) to the respective repo. |
| **How to avoid version conflicts?** | Run semantic_uncertainty and semantic-density in **separate environments** and expose them as **HTTP services** (or subprocess scripts). HASHIRU only needs `requests` and the service URL(s); it never imports the other repos. |
| **Config** | Set `HASHIRU_METRICS_SERVICE_URL` (and optionally `HASHIRU_METRICS_TIMEOUT`) so that `/score` is either the unified gateway or the single service that implements both metrics. |

The current `semantic_metrics.py` uses a single `/score` endpoint; the `service_wrapper` can be updated to call the two backend services (each in its own venv) and merge their results so that HASHIRU keeps a single URL.
