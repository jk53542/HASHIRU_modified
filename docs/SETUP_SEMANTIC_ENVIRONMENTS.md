# Setting Up Environments for Semantic Metrics

This guide walks you through creating separate environments for the **semantic uncertainty** (entropy) and **semantic density** repos, and clarifies where to run the FastAPI service wrapper.

---

## Do I need a separate environment for the FastAPI service wrapper?

**No.** You can run the FastAPI service wrapper in the **same environment as HASHIRU**.

The wrapper (`service_wrapper/comm_app.py`) only uses:

- FastAPI
- Pydantic

It does **not** import the semantic_uncertainty or semantic_density code. So it has no dependency conflict with HASHIRU. As long as your HASHIRU env has `fastapi`, `uvicorn`, and `pydantic` (and optionally `requests` for the client side), you can start the wrapper from the HASHIRU env.

**Summary:**

| Component              | Environment to use      |
|------------------------|-------------------------|
| HASHIRU app            | HASHIRU venv (already set up) |
| FastAPI service wrapper| **Same as HASHIRU** (no extra env) |
| Semantic entropy service | **New env:** semantic_uncertainty |
| Semantic density service | **New env:** semantic_density |

The two *backend* services (entropy and density) must run in their own environments because they use different versions of PyTorch, Transformers, etc.

---

## Prerequisites

- **Conda** (Miniconda or Anaconda) — both repos provide conda environment files. If you prefer **venv + pip**, optional steps are given for semantic_uncertainty; semantic_density is best installed via conda due to CUDA/pytorch.
- **WSL/Ubuntu** (or Linux) — the yaml files are Linux-oriented; on Windows-native you may need to adjust or use WSL.
- Enough disk space for two extra environments (each can be several GB due to PyTorch/CUDA).

---

## Part 1: Semantic uncertainty (entropy) environment

This repo uses **Python 3.11** and PyTorch with CUDA 11.8.

### Option A: Conda (recommended)

1. **Open a terminal** and go to the semantic_uncertainty repo:
   ```bash
   cd /home/knowltjo/hashiru_modified/semantic_uncertainty
   ```
   (Adjust the path if your project lives elsewhere.)

2. **Create the environment** from the source file (this is more portable than the full export):
   ```bash
   conda env create -f environment.yaml
   ```
   - Environment name will be `semantic_uncertainty` (from the `name:` in the file).
   - First run can take 10–20 minutes while conda resolves and installs packages.

3. **Activate it:**
   ```bash
   conda activate semantic_uncertainty
   ```

4. **Verify the stack:**
   ```bash
   python -c "import torch; print(torch.__version__)"
   python -c "import transformers; print(transformers.__version__)"
   ```
   You should see the versions conda installed (e.g. PyTorch 2.x, Transformers 4.x).

5. **Verify the entropy module can be imported:**  
   The repo uses top-level imports `from uncertainty.xxx`, so the directory that *contains* the `uncertainty` folder must be on `PYTHONPATH` (the **inner** `semantic_uncertainty` folder). From the repo root:
   ```bash
   cd /home/knowltjo/hashiru_modified/semantic_uncertainty
   export PYTHONPATH="$(pwd)/semantic_uncertainty:$PYTHONPATH"
   python -c "
   from uncertainty.uncertainty_measures.semantic_entropy import (
       get_semantic_ids, cluster_assignment_entropy
   )
   print('semantic_entropy imports OK')
   "
   ```
   If you see `semantic_entropy imports OK`, the environment is ready for building an entropy service.  
   **Tip:** For any script or service you run in this repo, set `PYTHONPATH` the same way (or add it to your shell profile when working in this env).

6. **Deactivate when done:**
   ```bash
   conda deactivate
   ```

### Option B: venv + pip (alternative)

Use this only if you cannot or do not want to use conda. You will need to align PyTorch/CUDA versions manually.

1. **Create and activate a venv** (Python 3.11 recommended):
   ```bash
   cd /home/knowltjo/hashiru_modified/semantic_uncertainty
   python3.11 -m venv .venv_entropy
   source .venv_entropy/bin/activate   # Linux/WSL
   ```

2. **Install PyTorch with CUDA 11.8** (match the conda env):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Install the rest** (from `environment.yaml` pip section):
   ```bash
   pip install "transformers>=4.31" evaluate datasets scikit-learn pandas scipy \
     wandb tokenizers accelerate omegaconf nltk tenacity sentencepiece safetensors \
     openai tiktoken einops ml_collections torchmetrics
   ```

4. **Install the package in editable mode** so imports like `semantic_uncertainty.uncertainty...` work:
   ```bash
   pip install -e .
   ```
   If the repo has no `setup.py` or `pyproject.toml`, set `PYTHONPATH` to the repo root when you run scripts:
   ```bash
   export PYTHONPATH=/home/knowltjo/hashiru_modified/semantic_uncertainty
   ```

5. Run the same import check as in step 5 of Option A.

---

## Part 2: Semantic density environment

This repo uses **Python 3.10** and a pinned set of packages (including PyTorch, Transformers, CUDA libs). The provided file is a **full export** with a `prefix:` that may not match your machine.

### Option A: Conda using the full export (if it works on your OS)

1. **Go to the semantic-density-paper repo:**
   ```bash
   cd /home/knowltjo/hashiru_modified/semantic-density-paper
   ```

2. **Create the environment:**
   ```bash
   conda env create -f environment_llama3.yml
   ```
   - Name will be `semantic_density_llama3`.
   - If creation fails (e.g. platform or prefix issues), use Option B.

3. **Activate:**
   ```bash
   conda activate semantic_density_llama3
   ```

4. **Verify:**
   ```bash
   python -c "import torch; print(torch.__version__)"
   python -c "import transformers; print(transformers.__version__)"
   ```

5. **Verify semantic_metrics (used for density):**  
   The repo uses `experiment_code/` as the working directory for imports (no package install). From repo root:
   ```bash
   cd experiment_code
   python -c "
   from semantic_metrics import (
       compute_semantic_density_from_similarity_matrix,
       compute_semantic_density_from_text_responses,
   )
   print('semantic_metrics imports OK')
   "
   ```
   Always run your density service from `semantic-density-paper/experiment_code` (or set `PYTHONPATH` to that folder).

5b. **For meaningful semantic density (recommended):** The density pipeline uses real sentence embeddings when `sentence-transformers` is available. Install it in the density env:
   ```bash
   pip install sentence-transformers
   ```
   Without it, the service falls back to fake embeddings and density values will be unreliable (often 0 or near 0).

6. **Deactivate:**
   ```bash
   conda deactivate
   ```

### Option B: Conda with a hand-written env file (no prefix)

If `environment_llama3.yml` fails (e.g. different Linux or prefix), create a minimal env that matches the repo’s needs:

1. **Create a new file** `environment_minimal.yml` in `semantic-density-paper/`:

   ```yaml
   name: semantic_density
   channels:
     - pytorch
     - defaults
   dependencies:
     - python=3.10
     - pip
     - pytorch
     - pytorch-cuda=11.8
     - numpy
     - scipy
     - pip:
         - transformers>=4.35
         - torch
         - scikit-learn
         - pandas
         - evaluate
         - datasets
         - tqdm
         - wandb
         - sentencepiece
         - safetensors
   ```

2. **Create and activate:**
   ```bash
   cd /home/knowltjo/hashiru_modified/semantic-density-paper
   conda env create -f environment_minimal.yml
   conda activate semantic_density
   ```

3. **Install any missing deps** the experiment scripts need (e.g. `rouge-score`, `nltk`) by running the script once and doing `pip install <package>` for import errors.

4. Run the same `semantic_metrics` import check as in Option A step 5 (from `experiment_code`).

---

## Part 3: Quick reference – where to run what

| What you run                | Command / env           |
|----------------------------|-------------------------|
| HASHIRU                    | Activate HASHIRU venv → `python app.py` (or your entrypoint) |
| FastAPI metrics wrapper    | **Same HASHIRU venv** → `uvicorn service_wrapper.comm_app:app --host 127.0.0.1 --port 8123` (from project root so `service_wrapper` is importable) |
| Entropy backend service    | See “Start entropy service” below (requires PYTHONPATH). |
| Density backend service   | `conda activate semantic_density` → `cd .../experiment_code` → `uvicorn density_service:app --host 127.0.0.1 --port 8125` |

**Ports matter:** the gateway calls **entropy** at 8124 and **density** at 8125. If you start entropy on 8125 and density on 8124, the gateway will still use the returned values (it detects responses by key), but for clarity run entropy on **8124** and density on **8125**.

### Start entropy service (port 8124)

The entropy app imports `from uncertainty.xxx`, so Python must see the directory that **contains** the `uncertainty` package (the **inner** `semantic_uncertainty` folder). In a new terminal, run:

```bash
conda activate semantic_uncertainty
cd /home/knowltjo/hashiru_modified/semantic_uncertainty/semantic_uncertainty
export PYTHONPATH="$(pwd)"
uvicorn entropy_service:app --host 127.0.0.1 --port 8124
```

If you get `ModuleNotFoundError: No module named 'uncertainty'`, you are not in the inner directory or PYTHONPATH is unset. Fix by `cd`-ing into the folder that contains the `uncertainty` subfolder and running `export PYTHONPATH="$(pwd)"` before uvicorn.

---

## Part 4: Optional – one-command activate helpers

You can add small scripts or shell aliases so you don’t have to remember paths.

**Example: start wrapper (from HASHIRU env)**  
In your HASHIRU venv, from the project root that contains `service_wrapper`:

```bash
# From hashiru_modified (parent of HASHIRU_modified and service_wrapper)
uvicorn service_wrapper.comm_app:app --host 127.0.0.1 --port 8123
```

**Example: activate entropy env and run a future entropy service**  
```bash
conda activate semantic_uncertainty
cd /home/knowltjo/hashiru_modified/semantic_uncertainty
# python your_entropy_service_app.py
```

**Example: activate density env and run a future density service**  
```bash
conda activate semantic_density_llama3
cd /home/knowltjo/hashiru_modified/semantic-density-paper/experiment_code
# python your_density_service_app.py
```

---

## Troubleshooting

### `ImportError: libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent` (semantic_uncertainty)

This happens when conda installs **MKL 2024.1+** together with PyTorch; the two are incompatible. Use one of these fixes **in the already-created `semantic_uncertainty` env**:

**Option A – Downgrade MKL (quick, keep conda PyTorch)**

```bash
conda activate semantic_uncertainty
conda install "mkl=2024.0.0"
python -c "import torch; print(torch.__version__)"
```

If conda refuses to downgrade (dependency conflict), try:

```bash
conda install mkl=2023.2.0 intel-openmp=2023.2.0
```

**Option B – Use PyTorch from pip instead of conda (avoids MKL mix)**

```bash
conda activate semantic_uncertainty
conda uninstall pytorch torchvision torchaudio pytorch-cuda --force
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print(torch.__version__)"
```

Use `cu118` for CUDA 11.8, or `cu121` for CUDA 12.1. For CPU-only: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`.

After either option, re-run the stack check and the `semantic_entropy` import test from Part 1.

### `iJIT_NotifyEvent` or `undefined symbol` in **semantic_density** (sentence_transformers)

If the density service (or `python -c "from sentence_transformers import ..."`) fails with `libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent`, PyTorch (conda or pip) is conflicting with Intel MKL on your system. Without sentence_transformers, density falls back to fake embeddings and **density values will be unreliable (often 0.0)**.

Try these in order:

**1. Environment variable (quick try)**  
Before starting the density service or running Python:

```bash
conda activate semantic_density
export MKL_DEBUG_CPU_TYPE=5
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('OK')"
```

If that prints `OK`, start the density service with the same env set:  
`MKL_DEBUG_CPU_TYPE=5 uvicorn density_service:app --host 127.0.0.1 --port 8125`

**2. PyTorch from conda-forge**  
Use a different PyTorch build that may avoid the bad MKL path:

```bash
conda activate semantic_density
pip uninstall torch torchvision torchaudio -y
conda install -c conda-forge pytorch cpuonly -y
pip install sentence-transformers
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('OK')"
```

**3. Clean pip reinstall (CPU wheel)**  
Force a fresh CPU-only PyTorch install:

```bash
conda activate semantic_density
pip uninstall torch torchvision torchaudio -y
pip cache purge
pip install torch --no-cache-dir --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('OK')"
```

**4. New env with only pip torch**  
If the conda env keeps pulling in a conflicting MKL, use a minimal venv:

```bash
cd ~/hashiru_modified/semantic-density-paper/experiment_code
python3 -m venv .venv_density
source .venv_density/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers numpy fastapi uvicorn
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('OK')"
```

Then start the density service from this venv (`source .venv_density/bin/activate` and run `uvicorn density_service:app ...`).

After any fix, restart the density service; you should see **"Using SentenceTransformer for semantic density embeddings"** instead of "using fake embeddings".

---

- **Conda solve is very slow or fails:** Try with mamba: `conda install mamba` then `mamba env create -f environment.yaml`.
- **CUDA not found:** Install a matching CUDA toolkit (e.g. 11.8 for semantic_uncertainty) or use CPU-only PyTorch for testing (`pip install torch --index-url https://download.pytorch.org/whl/cpu`).
- **Import errors for `semantic_uncertainty` or `semantic_metrics`:**  
  - **semantic_uncertainty:** The code imports `from uncertainty.xxx`, so `PYTHONPATH` must include the **inner** package directory (the folder that contains `uncertainty/`), e.g. `export PYTHONPATH=/path/to/semantic_uncertainty/semantic_uncertainty:$PYTHONPATH`. Then use `from uncertainty.uncertainty_measures.semantic_entropy import ...`.  
  - **semantic_density:** Run from `semantic-density-paper/experiment_code` or set `PYTHONPATH` to that folder.
- **`prefix` in yaml:** If the yaml contains `prefix: /some/path`, you can delete that line and run `conda env create -f ...` again so conda uses your default envs directory.

### Semantic density: `safetensors` / pip conflict (ResolutionImpossible)

If `conda env create -f environment_llama3.yml` fails with a pip conflict involving **safetensors** (e.g. `safetensors==0.4.0` vs `transformers` needing `safetensors>=0.4.1` and/or `auto-gptq`/`peft`), use the **minimal** environment instead. It avoids strict pins and packages you don’t need for the density service:

```bash
cd /home/knowltjo/hashiru_modified/semantic-density-paper
conda env create -f environment_minimal.yml
conda activate semantic_density
```

Then verify from `experiment_code` (see Part 2). The minimal env has everything needed for `semantic_metrics` and the DeBERTa-based density scripts; it does not install `auto-gptq` or `peft`. If you later need the full paper pipeline with those, create a separate env and relax the safetensors pin in the yaml to `safetensors>=0.4.1` (and consider dropping or loosening auto-gptq/peft if conflicts persist).

---

### HuggingFace requests and "loading weights"

When the entropy or density service starts, you will see many HTTP requests to `huggingface.co`. This is normal: the libraries are downloading the **NLI/embedding models** (DeBERTa, all-MiniLM-L6-v2). Some 404s (e.g. `adapter_config.json`) are optional; the services set `httpx` and `huggingface_hub` to WARNING to reduce noise. **"Loading weights"** = the NLI/embedding model, not your agent LLM. Set `HF_TOKEN` for faster downloads.

### Semantic entropy always 0.0

If **entropy = 0.0** always, the NLI model is putting all responses in one semantic cluster. You can try:

1. **`STRICT_ENTAILMENT=false`** – looser clustering (fewer pairs count as equivalent).
2. **`ENTAILMENT_THRESHOLD=0.9`** (or `0.95`) – use probability-based equivalence: two responses are in the same cluster only if both directions have P(entailment) ≥ this value. Higher values give more clusters. Set in the env where you start the entropy service, e.g. `export ENTAILMENT_THRESHOLD=0.95`.
3. **`ENTROPY_LAST_N_WORDS=200`** – use the last 200 words of each response instead of 100 so more variation is included (default 100). Use **4+ samples** per prompt (HASHIRU uses 4 by default).

---

Once both environments are set up, you can implement the two HTTP services (entropy and density) as described in `INTEGRATION_SEMANTIC_METRICS.md` and point HASHIRU’s `HASHIRU_METRICS_SERVICE_URL` at the wrapper or at the individual services.
