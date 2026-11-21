import re
import collections
from typing import Any, List, Set, Dict, Tuple
from pydantic import BaseModel, Field


class DatasetSample(BaseModel):
    id: str
    instruction: str
    output: str
    quality_score: float = 1.0


class MinHashLSH:
    """Locality-Sensitive Hashing (LSH) with MinHash for sub-quadratic O(N * K) deduplication."""

    def __init__(self, num_perm: int = 64, num_bands: int = 16, seed: int = 42):
        self.num_perm = num_perm
        self.num_bands = num_bands
        self.rows_per_band = num_perm // num_bands
        self.prime = 2147483647  # 2^31 - 1

        # Generate deterministic coefficients for universal hash permutations
        import random
        rng = random.Random(seed)
        self.a = [rng.randint(1, self.prime - 1) for _ in range(num_perm)]
        self.b = [rng.randint(0, self.prime - 1) for _ in range(num_perm)]

        # Buckets: (band_idx, band_hash) -> list[int]
        self.buckets: Dict[Tuple[int, int], List[int]] = collections.defaultdict(list)
        self.signatures: List[List[int]] = []

    def _tokenize(self, text: str) -> Set[int]:
        tokens = re.findall(r"\b\w+\b", text.lower())
        return {hash(tok) & 0xFFFFFFFF for tok in tokens}

    def compute_signature(self, text: str) -> List[int]:
        tokens = self._tokenize(text)
        if not tokens:
            return [0] * self.num_perm

        sig = []
        for i in range(self.num_perm):
            min_val = self.prime
            a_i = self.a[i]
            b_i = self.b[i]
            for tok in tokens:
                h = ((a_i * tok + b_i) % self.prime)
                if h < min_val:
                    min_val = h
            sig.append(min_val)
        return sig

    def is_duplicate(self, sig: List[int], threshold: float = 0.8) -> bool:
        """Checks whether the signature collides in any bucket with estimated similarity >= threshold."""
        candidate_indices: Set[int] = set()

        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_tuple = tuple(sig[start:end])
            band_hash = hash(band_tuple)

            for cand_idx in self.buckets.get((band_idx, band_hash), []):
                candidate_indices.add(cand_idx)

        # Compute estimated MinHash similarity for each candidate
        for cand_idx in candidate_indices:
            cand_sig = self.signatures[cand_idx]
            match_count = sum(1 for x, y in zip(sig, cand_sig) if x == y)
            est_similarity = match_count / self.num_perm
            if est_similarity >= threshold:
                return True

        return False

    def insert(self, sig: List[int], idx: int):
        self.signatures.append(sig)
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_tuple = tuple(sig[start:end])
            band_hash = hash(band_tuple)
            self.buckets[(band_idx, band_hash)].append(idx)


class DatasetCurator:
    """Enterprise Dataset Curator & Quality Filtering Engine.

    Applies sub-quadratic deduplication via MinHash LSH O(N * K), degenerate
    repetition detection (LLM hallucination loops), and quality filtering.
    """

    @staticmethod
    def jaccard_similarity(str1: str, str2: str) -> float:
        set1 = set(re.findall(r"\b\w+\b", str1.lower()))
        set2 = set(re.findall(r"\b\w+\b", str2.lower()))
        if not set1 or not set2:
            return 0.0
        return len(set1.intersection(set2)) / len(set1.union(set2))

    @staticmethod
    def detect_repetitive_loop(text: str, max_consecutive_repeats: int = 3) -> bool:
        """Detects consecutive n-gram repetition loops characteristic of model collapse."""
        words = re.findall(r"\b\w+\b", text.lower())
        if len(words) < 6:
            return False

        # Check for consecutive repetitions of word chunks of length L (1 to 8 words)
        for L in range(1, 9):
            consecutive = 1
            for i in range(L, len(words) - L + 1, L):
                chunk_prev = words[i - L : i]
                chunk_curr = words[i : i + L]
                if chunk_prev == chunk_curr:
                    consecutive += 1
                    if consecutive >= max_consecutive_repeats:
                        return True
                else:
                    consecutive = 1

        return False

    def deduplicate_and_filter(
        self,
        samples: list[DatasetSample],
        similarity_threshold: float = 0.8,
        min_output_length: int = 10,
        use_lsh: bool = True,
    ) -> list[DatasetSample]:
        clean_samples: list[DatasetSample] = []
        lsh = MinHashLSH(num_perm=64, num_bands=16) if use_lsh else None

        for sample in samples:
            # 1. Minimum output length filter
            if len(sample.output.strip()) < min_output_length:
                continue

            # 2. Anti-hallucination / degenerate loop filter
            if self.detect_repetitive_loop(sample.output):
                continue

            # 3. Duplicate detection
            if use_lsh and lsh is not None:
                sig = lsh.compute_signature(sample.instruction)
                if lsh.is_duplicate(sig, threshold=similarity_threshold):
                    continue
                lsh.insert(sig, len(clean_samples))
                clean_samples.append(sample)
            else:
                is_duplicate = False
                for existing in clean_samples:
                    sim = self.jaccard_similarity(sample.instruction, existing.instruction)
                    if sim >= similarity_threshold:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    clean_samples.append(sample)

        return clean_samples


# Singleton instance
dataset_curator = DatasetCurator()
