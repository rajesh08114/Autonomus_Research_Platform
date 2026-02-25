from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_ENV: str = "production"
    ALLOWED_ORIGINS: List[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    LOG_LEVEL: str = "INFO"

    MASTER_LLM_PROVIDER: str = "auto"
    MASTER_LLM_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
    MASTER_LLM_API_KEY: str = ""
    MASTER_LLM_MAX_TOKENS: int = 4096
    MASTER_LLM_TEMPERATURE: float = 0.1
    ALLOW_RULE_BASED_FALLBACK: bool = False
    HF_API_KEY: str = ""
    HF_INFERENCE_URL: str = "https://router.huggingface.co/v1"
    HF_MODEL_ID: str = "Qwen/Qwen2.5-7B-Instruct"

    QUANTUM_LLM_ENDPOINT: str = ""
    QUANTUM_LLM_API_KEY: str = ""
    QUANTUM_LLM_TIMEOUT: int = 120
    CODEHUB_USE_FOR_QUANTUM: bool = True
    CODEHUB_BACKEND_BASE_URL: str = ""
    CODEHUB_GENERATE_ENDPOINT: str = "/api/code/generate"
    CODEHUB_INTERNAL_API_KEY: str = ""
    CODEHUB_BEARER_TOKEN: str = ""
    CODEHUB_TIMEOUT: int = 120

    MAX_RETRY_COUNT: int = 5
    MAX_LLM_RETRIES: int = 3
    SUBPROCESS_TIMEOUT: int = 3600
    MAX_CONCURRENT_EXPS: int = 10
    MAX_STATE_SIZE_KB: int = 500
    STDOUT_CAP_CHARS: int = 10000
    STDERR_CAP_CHARS: int = 5000

    PROJECT_ROOT: str = "./workspace/projects"
    STATE_DB_PATH: str = "./workspace/state.db"
    KAGGLE_CONFIG_DIR: str = "~/.kaggle"

    QUANTUM_ENABLED: bool = True
    KAGGLE_ENABLED: bool = True
    GPU_ALLOWED: bool = False
    WEBHOOK_ENABLED: bool = False
    ENABLE_PACKAGE_INSTALL: bool = False
    EXPERIMENT_VENV_ENABLED: bool = True
    AUTO_CONFIRM_LOW_RISK: bool = True
    LOW_RISK_PACKAGES: str = "numpy,pandas,matplotlib,scikit-learn,requests"
    WORKFLOW_BACKGROUND_ENABLED: bool = True
    EXECUTION_MODE: str = "vscode_extension"
    LOCAL_PYTHON_COMMAND: str = "python"
    METRICS_TABLE_ENABLED: bool = True
    RL_ENABLED: bool = True
    RL_FEEDBACK_WINDOW: int = 200
    RL_MIN_SAMPLES_FOR_POLICY: int = 5

    FAILURE_INJECTION_ENABLED: bool = False
    FAILURE_INJECTION_RATE: float = 0.0
    FAILURE_INJECTION_POINTS: str = ""

    AUTO_RETRY_ON_LOW_METRIC: bool = True
    MIN_PRIMARY_METRIC_FOR_SUCCESS: float = 0.75
    CHAT_CONTEXT_LIMIT_DEFAULT: int = 5
    CHAT_CONTEXT_LIMIT_MAX: int = 20
    CHAT_HISTORY_LIMIT: int = 40

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    @property
    def project_root_path(self) -> Path:
        return Path(self.PROJECT_ROOT).expanduser().resolve()

    @property
    def state_db_path(self) -> Path:
        return Path(self.STATE_DB_PATH).expanduser().resolve()

    @property
    def huggingface_api_key(self) -> str:
        return self.HF_API_KEY or self.MASTER_LLM_API_KEY

    @property
    def huggingface_inference_url(self) -> str:
        return self.HF_INFERENCE_URL.rstrip("/")

    @property
    def huggingface_model_id(self) -> str:
        if self.HF_MODEL_ID:
            return self.HF_MODEL_ID
        if self.MASTER_LLM_MODEL:
            return self.MASTER_LLM_MODEL
        return "Qwen/Qwen2.5-7B-Instruct"

    @property
    def master_llm_provider_normalized(self) -> str:
        return str(self.MASTER_LLM_PROVIDER or "auto").strip().lower() or "auto"

    @property
    def effective_master_llm_provider(self) -> str:
        provider = self.master_llm_provider_normalized
        if provider in {"huggingface", "hf", "hugging_face"}:
            if bool(self.huggingface_api_key):
                return "huggingface"
            return "rule_based" if self.ALLOW_RULE_BASED_FALLBACK else "huggingface"
        if provider == "rule_based":
            return "rule_based" if self.ALLOW_RULE_BASED_FALLBACK else "huggingface"
        if provider == "auto":
            if bool(self.huggingface_api_key):
                return "huggingface"
            return "rule_based" if self.ALLOW_RULE_BASED_FALLBACK else "huggingface"
        return "huggingface"

    @property
    def codehub_base_url(self) -> str:
        return str(self.CODEHUB_BACKEND_BASE_URL or "").strip().rstrip("/")

    @property
    def codehub_generate_url(self) -> str:
        base = self.codehub_base_url
        if not base:
            return ""
        endpoint = "/" + str(self.CODEHUB_GENERATE_ENDPOINT or "/api/code/generate").lstrip("/")
        return f"{base}{endpoint}"

    @property
    def codehub_enabled(self) -> bool:
        return bool(self.CODEHUB_USE_FOR_QUANTUM and self.codehub_generate_url)


settings = Settings()
