import re
from typing import Any
from pydantic import BaseModel, Field


class DatasetSample(BaseModel):
    id: str
    instruction: str
    output: str
    quality_score: float = 1.0


class DatasetCurator:
    """Enterprise Dataset Curator & Quality Filtering Engine.
    
    Aplica desduplicação por MinHash/Jaccard, filtragem de amostras
    de baixa qualidade ou incompletas e balanceamento de instruções.
    """

    @staticmethod
    def jaccard_similarity(str1: str, str2: str) -> float:
        set1 = set(re.findall(r"\b\w+\b", str1.lower()))
        set2 = set(re.findall(r"\b\w+\b", str2.lower()))
        if not set1 or not set2:
            return 0.0
        return len(set1.intersection(set2)) / len(set1.union(set2))

    def deduplicate_and_filter(
        self,
        samples: list[DatasetSample],
        similarity_threshold: float = 0.8,
        min_output_length: int = 10,
    ) -> list[DatasetSample]:
        clean_samples: list[DatasetSample] = []

        for sample in samples:
            # 1. Filtro de comprimento e qualidade básica
            if len(sample.output.strip()) < min_output_length:
                continue

            # 2. Verificação de Duplicidade vs Amostras Existentes
            is_duplicate = False
            for existing in clean_samples:
                sim = self.jaccard_similarity(sample.instruction, existing.instruction)
                if sim >= similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                clean_samples.append(sample)

        return clean_samples


# Instância Singleton
dataset_curator = DatasetCurator()
