import json
import os
from typing import List, Dict, Any
from app.generators.evolver import EvolInstructMutator


class SyntheticDataPipeline:
    """Synthetic dataset generation pipeline for LLM fine-tuning."""

    SEED_PROMPTS = [
        {"domain": "Customer Support", "instruction": "How should we respond to a customer dissatisfied with a delivery delay?"},
        {"domain": "DevOps", "instruction": "Explain how to configure a Docker Compose healthcheck for FastAPI."},
        {"domain": "Finance", "instruction": "How do you calculate a prorated refund for subscription cancellations?"},
        {"domain": "Security", "instruction": "How do you sanitize and mask PII (such as SSNs or tax IDs) in prompts sent to LLMs?"}
    ]

    def __init__(self) -> None:
        self.mutator = EvolInstructMutator()

    def generate_dataset(self, num_samples_per_seed: int = 2) -> List[Dict[str, Any]]:
        """
        Generates a synthetic instruction-response dataset.
        """
        dataset = []
        sample_id = 1

        for seed in self.SEED_PROMPTS:
            for _ in range(num_samples_per_seed):
                mutation = self.mutator.mutate_prompt(seed["instruction"])
                synthetic_entry = {
                    "id": f"syn_{sample_id:03d}",
                    "domain": seed["domain"],
                    "instruction": mutation["evolved_instruction"],
                    "tactic": mutation["tactic_used"],
                    "input_context": f"Internal reference documentation for the {seed['domain']} domain.",
                    "response": f"Structured synthetic response generated to address the complex instruction '{seed['instruction']}' with a focus on accuracy and compliance."
                }
                dataset.append(synthetic_entry)
                sample_id += 1

        return dataset

    def export_dataset_json(self, dataset: List[Dict[str, Any]], output_path: str) -> str:
        """Exports the synthetic dataset in JSON or JSONL format for training."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        return output_path
