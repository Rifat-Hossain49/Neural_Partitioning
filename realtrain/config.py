from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass(frozen=True)
class Config:
    protocol_id: str = "waharp_realtrain_v1_three_domain_pageio_20260829"
    seed: int = 2026082951
    capacity: int = 128
    # Default scale chosen to fit one 12h Kaggle session on 2xT4 while still
    # training from all three real domains. Arizona is kept full by default.
    twitter_max_rows: int = 1_000_000
    crimes_max_rows: int = 1_000_000
    arizona_max_rows: int = 0  # 0 = all rows
    # Workload sizes.
    construction_query_count: int = 2800
    validation_queries_per_range_condition: int = 100
    validation_point_queries: int = 100
    validation_knn_queries_per_k: int = 100
    final_queries_per_range_condition: int = 300
    final_point_queries: int = 300
    final_knn_queries_per_k: int = 300
    # State-supervision budget.
    initial_states_per_domain: int = 8000
    dagger_states_per_domain: int = 2000
    candidates_per_state: int = 80
    state_min_pages: int = 2
    state_max_pages: int = 16
    local_query_cap: int = 128
    hist_bins: int = 12
    # Training.
    bootstrap_epochs: int = 12
    final_epochs: int = 28
    train_batch_size: int = 256
    learning_rate: float = 6e-4
    weight_decay: float = 1e-4
    ensemble_members: int = 3
    early_stop_patience: int = 6
    shadow_rows_per_domain: int = 250_000
    # Teacher loss components. Query-page hits are primary.
    teacher_overlap_weight: float = 1e-4
    teacher_margin_weight: float = 2e-5
    # Time control: leave 30 minutes before Kaggle's 12h hard stop.
    wall_budget_seconds: int = 41_400
    reserve_seconds: int = 1_800
    # Baselines.
    run_platon: bool = True
    run_tgs: bool = True
    run_str: bool = True
    run_guttman_native: bool = True
    run_rstar_native: bool = True
    platon_rollouts: int = 25
    platon_simulation_steps: int = 100
    platon_utilization: float = 0.8
    storage_page_bytes: int = 4096
    # Dynamic libspatialindex baselines.
    native_fill_factor: float = 0.7
    # Statistics.
    bootstrap_draws: int = 1000

    def to_dict(self):
        return asdict(self)

    def dump(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


DEFAULT_CONFIG = Config()
