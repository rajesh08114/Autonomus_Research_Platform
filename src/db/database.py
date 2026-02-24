from __future__ import annotations

import sqlite3
from pathlib import Path

from src.config.settings import settings


def get_connection() -> sqlite3.Connection:
    db_path = settings.state_db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


async def init_db() -> None:
    conn = get_connection()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id                  TEXT PRIMARY KEY,
                status              TEXT NOT NULL,
                phase               TEXT NOT NULL,
                prompt              TEXT NOT NULL,
                requires_quantum    BOOLEAN DEFAULT FALSE,
                quantum_framework   TEXT,
                framework           TEXT,
                dataset_source      TEXT,
                hardware_target     TEXT DEFAULT 'cpu',
                target_metric       TEXT DEFAULT 'accuracy',
                random_seed         INTEGER DEFAULT 42,
                retry_count         INTEGER DEFAULT 0,
                llm_calls_count     INTEGER DEFAULT 0,
                total_tokens_used   INTEGER DEFAULT 0,
                project_path        TEXT,
                documentation_path  TEXT,
                state_json          TEXT,
                created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
                completed_at        DATETIME
            );

            CREATE TABLE IF NOT EXISTS experiment_logs (
                id              TEXT PRIMARY KEY,
                experiment_id   TEXT,
                phase           TEXT,
                level           TEXT,
                message         TEXT,
                details_json    TEXT,
                timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS experiment_metrics (
                id              TEXT PRIMARY KEY,
                experiment_id   TEXT,
                metric_name     TEXT,
                metric_value    REAL,
                metric_type     TEXT,
                recorded_at     DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS denial_records (
                id                   TEXT PRIMARY KEY,
                experiment_id        TEXT,
                action               TEXT,
                denied_item          TEXT,
                reason               TEXT,
                alternative_offered  TEXT,
                alternative_accepted BOOLEAN DEFAULT FALSE,
                timestamp            DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS repair_history (
                id              TEXT PRIMARY KEY,
                experiment_id   TEXT,
                attempt         INTEGER,
                error_category  TEXT,
                fix_description TEXT,
                file_changed    TEXT,
                success         BOOLEAN,
                timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS rl_feedback (
                id              TEXT PRIMARY KEY,
                experiment_id   TEXT,
                phase           TEXT,
                reward          REAL,
                signal          TEXT,
                details_json    TEXT,
                timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS llm_usage (
                id                  TEXT PRIMARY KEY,
                experiment_id       TEXT,
                phase               TEXT,
                provider            TEXT,
                model               TEXT,
                latency_ms          REAL,
                prompt_tokens       INTEGER DEFAULT 0,
                completion_tokens   INTEGER DEFAULT 0,
                total_tokens        INTEGER DEFAULT 0,
                estimated_cost_usd  REAL DEFAULT 0,
                success             BOOLEAN DEFAULT TRUE,
                error_message       TEXT,
                timestamp           DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
            CREATE INDEX IF NOT EXISTS idx_experiments_phase ON experiments(phase);
            CREATE INDEX IF NOT EXISTS idx_logs_experiment ON experiment_logs(experiment_id);
            CREATE INDEX IF NOT EXISTS idx_logs_phase ON experiment_logs(phase);
            CREATE INDEX IF NOT EXISTS idx_metrics_experiment ON experiment_metrics(experiment_id);
            CREATE INDEX IF NOT EXISTS idx_rl_phase ON rl_feedback(phase);
            CREATE INDEX IF NOT EXISTS idx_rl_experiment ON rl_feedback(experiment_id);
            CREATE INDEX IF NOT EXISTS idx_llm_usage_experiment ON llm_usage(experiment_id);
            CREATE INDEX IF NOT EXISTS idx_llm_usage_provider ON llm_usage(provider);
            """
        )
        conn.commit()
    finally:
        conn.close()


def workspace_root() -> Path:
    return settings.project_root_path.parent
