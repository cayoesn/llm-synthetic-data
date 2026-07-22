import json
import os
from typing import List, Dict, Any
from app.generators.evolver import EvolInstructMutator


class SyntheticDataPipeline:
    """Pipeline de Geração de Datasets Sintéticos para Fine-Tuning."""

    SEED_PROMPTS = [
        {"domain": "Atendimento", "instruction": "Como responder a um cliente insatisfeito com atraso de entrega?"},
        {"domain": "DevOps", "instruction": "Explique como configurar um healthcheck no Docker Compose para FastAPI."},
        {"domain": "Finanças", "instruction": "Como calcular o reembolso proporcional em cancelamentos de assinatura?"},
        {"domain": "Segurança", "instruction": "Como higienizar e mascarar CPFs em prompts enviados para LLMs?"}
    ]

    def __init__(self) -> None:
        self.mutator = EvolInstructMutator()

    def generate_dataset(self, num_samples_per_seed: int = 2) -> List[Dict[str, Any]]:
        """
        Gera um dataset sintético instrução-resposta.
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
                    "input_context": f"Documentacao de referencia interna para o dominio {seed['domain']}.",
                    "response": f"Resposta estruturada sintetica gerada para atender a instrucao complexa '{seed['instruction']}' com foco em precisao e conformidade."
                }
                dataset.append(synthetic_entry)
                sample_id += 1

        return dataset

    def export_dataset_json(self, dataset: List[Dict[str, Any]], output_path: str) -> str:
        """Exporta o dataset sintético em formato JSON ou JSONL para treino."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        return output_path
