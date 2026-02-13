import traceback
__all__ = ["SemanticMetricsTool"]

class SemanticMetricsTool:
    dependencies = []   # @@ REQUIRED by HASHIRU's tool loader
    inputSchema = {
        "name": "compute_semantic_metrics",
        "description": "Compute semantic entropy and semantic density for a (prompt, response) pair.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The original user prompt or context."
                },
                "response": {
                    "type": "string",
                    "description": "The response produced by an agent."
                },
                "samples": {
                    "type": "string",
                    "description": "Optional comma-separated list of alternative responses for the same prompt. If omitted, pass agent_name to auto-generate samples."
                },
                "agent_name": {
                    "type": "string",
                    "description": "Optional. When provided and samples is omitted, the tool will ask this agent for extra responses (same prompt) to use as samples for meaningful entropy/density."
                },
                "mode": {
                    "type": "string",
                    "enum": ["fast", "balanced", "thorough"],
                    "default": "balanced",
                    "description": "Sampling mode for semantic metrics computation."
                }
            },
            "required": ["prompt", "response"]
        }
    }


    def __init__(self):
        from src.metrics.semantic_metrics import (
            compute_semantic_metrics_both,
            compute_semantic_entropy,
            compute_semantic_density,
        )
        self.compute_semantic_metrics_both = compute_semantic_metrics_both
        self.compute_semantic_entropy = compute_semantic_entropy
        self.compute_semantic_density = compute_semantic_density

    def run(self, **kwargs):
        """
        Properly call the underlying semantic metric functions.
        When samples is omitted, pass agent_name to auto-gather multiple responses for the same prompt.
        Framework calls this as tool.run(**query); accept kwargs for compatibility.
        """
        import traceback

        prompt = kwargs.get("prompt")
        response = kwargs.get("response")
        samples = kwargs.get("samples")
        agent_name = kwargs.get("agent_name")
        mode = kwargs.get("mode", "balanced")

        if prompt is None or response is None:
            return {
                "status": "error",
                "message": "Tool `compute_semantic_metrics` requires 'prompt' and 'response'.",
                "output": {"error": "missing prompt or response"}
            }

        try:
            # Convert samples into a list (if provided)
            if samples:
                samples_list = [s.strip() for s in samples.split(",") if s.strip()]
            else:
                samples_list = None

            # If no samples but agent_name given, gather extra responses from that agent for the same prompt
            if (not samples_list or len(samples_list) == 0) and agent_name:
                try:
                    from src.manager.agent_manager import AgentManager
                    _num_extra = 3  # number of extra responses for meaningful entropy/density
                    extra = AgentManager().get_agent_responses(agent_name, prompt, _num_extra)
                    if extra:
                        samples_list = extra
                        print(f"\n[SemanticMetricsTool] Auto-gathered {len(samples_list)} sample(s) from agent '{agent_name}'.\n")
                except Exception as e:
                    print(f"\n[SemanticMetricsTool] Could not gather samples from agent '{agent_name}': {e}\n")

            # Single gateway call so both entropy and density come from the same response (avoids timeout on first call while DeBERTa loads)
            print("\n[SemanticMetricsTool] Computing entropy and density (single request)...\n")
            both = self.compute_semantic_metrics_both(
                prompt=prompt,
                response=response,
                samples=samples_list
            )
            entropy_val = both.get("entropy", None)
            density_val = both.get("density", None)
            diag = both.get("diagnostics") or {}
            diag_e = diag
            diag_d = diag
            # Log raw gateway response for debugging (entropy/density 0.0 and 0.5 often mean timeout or backend error)
            print(f"\n[SemanticMetricsTool] Gateway response: entropy={entropy_val}, density={density_val}, elapsed_s={diag.get('elapsed_s')}, entropy_ok={diag.get('entropy_ok')}, density_ok={diag.get('density_ok')}\n")
            if diag.get("entropy_error") or diag.get("density_error"):
                print(f"\n[SemanticMetricsTool] Backend errors: entropy_error={diag.get('entropy_error')}, density_error={diag.get('density_error')}\n")

            n_used = len(samples_list) if samples_list else 0
            print(f"\n[SemanticMetricsTool] Entropy: {entropy_val}  (0 = all responses in one semantic cluster)\n")
            print(f"\n[SemanticMetricsTool] Density: {density_val}  (always in [0, 1]; {n_used} sample(s) used)\n")

            # Sanity check: warn if backends failed or no samples were used
            if not diag_e.get("entropy_ok", True) or not diag_d.get("density_ok", True):
                print("\n[SemanticMetricsTool] WARNING: One or both backends failed. Check that entropy (8124) and density (8125) services are running.\n")
            if not (samples_list and len(samples_list) > 0):
                print("\n[SemanticMetricsTool] NOTE: No samples were used. For meaningful metrics, call with 'agent_name' (to auto-gather samples) or pass 'samples' as comma-separated alternative responses.\n")

            print("\n[SemanticMetricsTool] SUCCESS\n")

            return {
                "status": "success",
                "message": "Semantic metrics computed",
                "output": {
                    "semantic_entropy": entropy_val,
                    "semantic_density": density_val,
                    "raw_entropy": both.get("entropy"),
                    "raw_density": both.get("density"),
                    "diagnostics": {
                        "entropy_ok": diag.get("entropy_ok", True),
                        "density_ok": diag.get("density_ok", True),
                        "used_samples": diag.get("used_samples", False),
                        "entropy_error": diag.get("entropy_error"),
                        "density_error": diag.get("density_error"),
                        "elapsed_s": diag.get("elapsed_s"),
                    }
                }
            }

        except Exception as e:
            print("\n[SemanticMetricsTool] FAILED\n")
            traceback.print_exc()

            return {
                "status": "error",
                "message": f"Tool `compute_semantic_metrics` failed: {str(e)}",
                "output": {
                    "error": str(e)
                }
            }

    # def run(self, **kwargs):
    #     prompt = kwargs.get("prompt")
    #     response = kwargs.get("response")
    #     samples = kwargs.get("samples")
    #     if samples_raw:
    #         samples = [s.strip() for s in samples_raw.split(",")]
    #     else:
    #         samples = None
    #     mode = kwargs.get("mode", "balanced")

    #     # Map mode -> sampling depth
    #     if mode == "fast":
    #         s_kwargs = {"samples": samples, "M": 2}
    #     elif mode == "thorough":
    #         s_kwargs = {"samples": samples, "M": 10}
    #     else:
    #         s_kwargs = {"samples": samples, "M": 4}

    #     entropy = self.compute_semantic_entropy(prompt=prompt, response=response, **s_kwargs)
    #     density = self.compute_semantic_density(prompt=prompt, response=response, **s_kwargs)

    #     return {
    #         "entropy": entropy.get("entropy"),
    #         "entropy_info": entropy.get("clusters"),
    #         "density": density.get("density"),
    #         "density_info": density.get("kernel_values"),
    #         "timings": {
    #             "entropy_s": entropy["diagnostics"]["elapsed_s"],
    #             "density_s": density["diagnostics"]["elapsed_s"]
    #         }
    #     }
