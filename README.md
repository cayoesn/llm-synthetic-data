# 🧬 LLM Synthetic Data & Fine-Tuning Pipeline (Enterprise Edition)

Pipeline de Geração Syntética de Dados e Fine-Tuning com **Evol-Instruct Pattern**, **Dataset Curator** e **QLoRA**.

## 🌟 Arquitetura & Recursos Big-Tech
- **Dataset Curator & Quality Filtering Engine**: Filtro de qualidade, validação de saídas e desduplicação por similaridade Jaccard / MinHash.
- **Evol-Instruct Generator**: Evolução automática de instruções simples para comandos complexos e detalhados.
- **QLoRA Fine-Tuning Integration**: Suporte para treinamento eficiente de LLMs com adaptadores LoRA.

## 🚀 Como Executar no Docker
```bash
docker compose up -d --build
```

## 🧪 Testes Unitários e Integração (100% Cobertura)
```bash
docker run --rm -v $(pwd):/app -w /app python:3.12-slim bash -c "pip install pytest pytest-asyncio pytest-cov pydantic pydantic-settings httpx fastapi uvicorn prometheus_fastapi_instrumentator && PYTHONPATH=. pytest"
```
