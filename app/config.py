from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurações Globais da Gerador de Dados Sintéticos e Fine-Tuning."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    APP_NAME: str = "LLM Synthetic Data & Fine-Tuning Suite"
    ENV: str = "development"

    # Parâmetros de Evolução de Prompts (Evol-Instruct)
    MUTATION_DEPTH: int = 3
    TARGET_DATASET_SIZE: int = 50

    # Configurações de Fine-Tuning QLoRA
    LORA_R: int = 16
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0.05
    LEARNING_RATE: float = 2e-4
    NUM_EPOCHS: int = 3

    # Langfuse Integration
    LANGFUSE_PUBLIC_KEY: str = "pk-lf-demo"
    LANGFUSE_SECRET_KEY: str = "sk-lf-demo"
    LANGFUSE_HOST: str = "http://localhost:3000"


settings = Settings()
