from __future__ import annotations

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(String, primary_key=True)
    status = Column(String, nullable=False)
    phase = Column(String, nullable=False)
    prompt = Column(Text, nullable=False)
    requires_quantum = Column(Boolean, default=False)
    quantum_framework = Column(String)
    framework = Column(String)
    dataset_source = Column(String)
    hardware_target = Column(String, default="cpu")
    target_metric = Column(String, default="accuracy")
    random_seed = Column(Integer, default=42)
    retry_count = Column(Integer, default=0)
    llm_calls_count = Column(Integer, default=0)
    total_tokens_used = Column(Integer, default=0)
    project_path = Column(String)
    documentation_path = Column(String)
    state_json = Column(Text)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    completed_at = Column(DateTime)


class ExperimentLog(Base):
    __tablename__ = "experiment_logs"

    id = Column(String, primary_key=True)
    experiment_id = Column(String, nullable=False)
    phase = Column(String)
    level = Column(String)
    message = Column(Text)
    details_json = Column(Text)
    timestamp = Column(DateTime)


class ExperimentMetric(Base):
    __tablename__ = "experiment_metrics"

    id = Column(String, primary_key=True)
    experiment_id = Column(String, nullable=False)
    metric_name = Column(String, nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_type = Column(String, nullable=False)
    recorded_at = Column(DateTime)

