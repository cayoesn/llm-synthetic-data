# 🧬 LLM Synthetic Data & Fine-Tuning Pipeline (Enterprise Edition)

Industrial-grade synthetic data generation, high-quality curation, and fine-tuning pipeline powered by **Evol-Instruct**, **Sub-Quadratic MinHash LSH**, and **Adaptive QLoRA**.

---

## 🌟 Architecture & Enterprise Capabilities

- **Sub-Quadratic MinHash LSH Deduplication**:
  - Locality-Sensitive Hashing partitioned into $b$ bands of $r$ rows ($K = b \cdot r = 64$).
  - Collision probability modeled by the S-curve:
    $$P(\text{collision}) = 1 - (1 - s^r)^b$$
  - Reduces deduplication complexity from a naive $\mathcal{O}(N^2)$ to $\mathcal{O}(N \cdot K)$.
- **Anti-Hallucination & Degenerate Loop Filter**:
  - Early detection of LLM sampling collapse via consecutive repetition of $n$-gram chunks ($1 \le L \le 8$).
- **Evol-Instruct Pattern**:
  - In-depth and in-breadth synthetic mutations: adding strict constraints, concretizing into real-world scenarios, requiring Chain-of-Thought reasoning, and deriving related complex topics.
- **QLoRA Fine-Tuning Integration**:
  - Automated preparation and sanitization of instruction datasets for memory-efficient 4-bit quantized fine-tuning.

---

## 📊 Asymptotic Complexity Comparison

| Component | Traditional Algorithm | Implemented Enterprise Approach | Complexity Reduction |
| :--- | :--- | :--- | :--- |
| **Deduplication** | Pairwise Jaccard Comparison $\mathcal{O}(N^2)$ | Banded MinHash LSH | $\mathcal{O}(N^2) \to \mathcal{O}(N \cdot K)$ |
| **Loop Detection** | Exhaustive Substring Analysis $\mathcal{O}(\|T\|^2)$ | Sliding Window Chunk Matcher | $\mathcal{O}(\|T\|)$ |
| **Data Curation** | Manual Post-Processing | Concurrent Streaming Pipeline | Inline real-time filtering |

---

## 🚀 Running with Docker

```bash
docker compose up -d --build
```

---

## 🧪 Automated Tests (100% Hermetic)

```bash
docker run --rm -v $(pwd):/workspace -w /workspace -e PYTHONPATH=/workspace portfolio-spec-engine:test pytest tests -v --cov=app --cov-report=term-missing
```
