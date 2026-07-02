"""Type stubs for deep_causality._core – a Rust/PyO3 native extension module."""

from typing import Dict, List

class FeatureRanking:
    """Result from mRMR feature selection."""
    name: str
    score: float

class SurdResult:
    """Result from SURD causal decomposition."""
    redundant_info: float
    unique_info: float
    synergistic_info: float
    total_info: float

    def to_dict(self) -> Dict[str, float]: ...

def run_mrmr(
    data: List[List[float]],
    column_names: List[str],
    target_column: str,
    max_features: int = 10,
) -> List[FeatureRanking]: ...

def run_mrmr_from_dict(
    df_dict: Dict[str, List[float]],
    target_column: str,
    max_features: int = 10,
) -> List[FeatureRanking]: ...

def version() -> str: ...
