# 🧬 LLM Synthetic Data & Fine-Tuning Pipeline (Enterprise Edition)

Pipeline Industrial de Geração Sintética de Dados, Curadoria de Alta Qualidade e Fine-Tuning baseado em **Evol-Instruct**, **MinHash LSH Sub-Quadrático** e **QLoRA Adaptativo**.

---

## 🌟 Arquitetura & Recursos Big-Tech

- **Desduplicação Sub-Quadrática MinHash LSH**:
  - Locality-Sensitive Hashing particionado em $b$ bandas de $r$ linhas ($K = b \cdot r = 64$).
  - Probabilidade de colisão modelada pela curva S:
    $$P(\text{colisão}) = 1 - (1 - s^r)^b$$
  - Reduz a complexidade de deduplicação de um ingênuo $\mathcal{O}(N^2)$ para $\mathcal{O}(N \cdot K)$.
- **Filtro Anti-Alucinação & Loop Degenerado**:
  - Detecção precoce de colapso de amostragem de LLMs via repetição consecutiva de blocos de $n$-gramas ($1 \le L \le 8$).
- **Evol-Instruct Pattern**:
  - Mutações sintéticas in-depth e in-breadth: adição de restrições estritas, concretização em cenários reais, exigência de Chain-of-Thought e derivação lateral de tópicos.
- **Integração QLoRA Fine-Tuning**:
  - Preparação e sanitização automatizada de datasets instrucionais para fine-tuning eficiente com quantização de 4 bits.

---

## 📊 Tabela de Complexidade Assintótica

| Componente | Algoritmo Tradicional | Abordagem Enterprise Implementada | Redução de Complexidade |
| :--- | :--- | :--- | :--- |
| **Deduplicação** | Comparação Pareada Jaccard $\mathcal{O}(N^2)$ | MinHash LSH com Banding | $\mathcal{O}(N^2) \to \mathcal{O}(N \cdot K)$ |
| **Loop Check** | Análise Exaustiva de Substrings $\mathcal{O}(|T|^2)$ | Sliding Window Chunk Matcher | $\mathcal{O}(|T|)$ |
| **Curadoria de Dados** | Pós-processamento Manual | Pipeline Streaming Concorrente | Filtragem instantânea inline |

---

## 🚀 Como Executar no Docker

```bash
docker compose up -d --build
```

---

## 🧪 Testes Automatizados (100% Hermético)

```bash
docker run --rm -v $(pwd):/workspace -w /workspace -e PYTHONPATH=/workspace portfolio-spec-engine:test pytest tests -v --cov=app --cov-report=term-missing
```
