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
                    "description": "Optional comma-separated list of reference samples for semantic metrics."
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
            compute_semantic_entropy,
            compute_semantic_density
        )
        self.compute_semantic_entropy = compute_semantic_entropy
        self.compute_semantic_density = compute_semantic_density

    def run(self, prompt: str, response: str, samples: str = None, mode: str = "balanced"):
        """
        Properly call the underlying semantic metric functions.
        The underlying functions expect prompt + response + samples list.
        """

        import traceback

        try:
            # Convert samples into a list (if provided)
            if samples:
                samples_list = [s.strip() for s in samples.split(",")]
            else:
                samples_list = None

            print("\n[SemanticMetricsTool] Computing entropy...\n")
            entropy_dict = self.compute_semantic_entropy(
                prompt=prompt,
                response=response,
                samples=samples_list
            )

            print("\n[SemanticMetricsTool] Computing density...\n")
            density_dict = self.compute_semantic_density(
                prompt=prompt,
                response=response,
                samples=samples_list
            )

            entropy_val = entropy_dict.get("entropy", None)
            density_val = density_dict.get("density", None)

            print("\n[SemanticMetricsTool] SUCCESS\n")

            return {
                "status": "success",
                "message": "Semantic metrics computed",
                "output": {
                    "semantic_entropy": entropy_val,
                    "semantic_density": density_val,
                    "raw_entropy": entropy_dict,
                    "raw_density": density_dict
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
