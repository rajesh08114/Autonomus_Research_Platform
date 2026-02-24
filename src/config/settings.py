from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_ENV: str = "production"
    ALLOWED_ORIGINS: List[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    LOG_LEVEL: str = "INFO"

    MASTER_LLM_PROVIDER: str = "rule_based"
    MASTER_LLM_MODEL: str = "deterministic-orchestrator"
    MASTER_LLM_API_KEY: str = ""
    MASTER_LLM_MAX_TOKENS: int = 4096
    MASTER_LLM_TEMPERATURE: float = 0.1
    HF_API_KEY: str = ""
    HF_INFERENCE_URL: str = "https://router.huggingface.co/v1"
    HF_MODEL_ID: str = "Qwen/Qwen2.5-7B-Instruct"

    QUANTUM_LLM_ENDPOINT: str = ""
    QUANTUM_LLM_API_KEY: str = ""
    QUANTUM_LLM_TIMEOUT: int = 120

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
    METRICS_TABLE_ENABLED: bool = True
    RL_ENABLED: bool = True
    RL_FEEDBACK_WINDOW: int = 200
    RL_MIN_SAMPLES_FOR_POLICY: int = 5

    LLM_COST_PER_1K_INPUT_TOKENS: float = 0.0
    LLM_COST_PER_1K_OUTPUT_TOKENS: float = 0.0

    FAILURE_INJECTION_ENABLED: bool = False
    FAILURE_INJECTION_RATE: float = 0.0
    FAILURE_INJECTION_POINTS: str = ""

    AUTO_RETRY_ON_LOW_METRIC: bool = True
    MIN_PRIMARY_METRIC_FOR_SUCCESS: float = 0.75

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
        if self.MASTER_LLM_MODEL and self.MASTER_LLM_MODEL != "deterministic-orchestrator":
            return self.MASTER_LLM_MODEL
        return "Qwen/Qwen2.5-7B-Instruct"


settings = Settings()
