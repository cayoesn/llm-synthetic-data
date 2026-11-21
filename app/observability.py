import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class SyntheticObservability:
    """Langfuse observability for synthetic datasets and QLoRA fine-tuning runs."""

    def __init__(self) -> None:
        self.client = None
        self._init_client()

    def _init_client(self) -> None:
        try:
            from langfuse import Langfuse
            from app.config import settings
            self.client = Langfuse(
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY,
                host=settings.LANGFUSE_HOST
            )
            logger.info("Langfuse client initialized in the Synthetic Data Suite.")
        except Exception as e:
            logger.warning(f"Langfuse SDK unavailable: {e}")
            self.client = None

    def trace_dataset_generation(self, num_samples: int, domains: List[str]) -> Optional[str]:
        if not self.client:
            return None
        try:
            if hasattr(self.client, "trace"):
                trace = self.client.trace(
                    name="synthetic_dataset_generation",
                    metadata={"num_samples": num_samples, "domains": domains},
                    tags=["synthetic_data", "evol_instruct"]
                )
                return getattr(trace, "id", "demo-dataset-trace")
            return "demo-dataset-trace"
        except Exception as e:
            logger.error(f"Failed to log dataset generation trace: {e}")
            return None

    def trace_finetuning_run(self, train_summary: Dict[str, Any]) -> Optional[str]:
        if not self.client:
            return None
        try:
            if hasattr(self.client, "trace"):
                trace = self.client.trace(
                    name="qlora_finetuning_run",
                    metadata=train_summary,
                    tags=["fine_tuning", "qlora", "peft"]
                )
                if hasattr(trace, "score"):
                    trace.score(
                        name="final_training_loss",
                        value=float(train_summary.get("final_loss", 0.0))
                    )
                return getattr(trace, "id", "demo-training-trace")
            return "demo-training-trace"
        except Exception as e:
            logger.error(f"Failed to log fine-tuning trace: {e}")
            return None


synth_obs = SyntheticObservability()
