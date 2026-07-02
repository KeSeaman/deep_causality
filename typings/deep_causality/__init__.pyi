"""Type stubs for the deep_causality package."""

from deep_causality._core import (
    FeatureRanking as FeatureRanking,
    SurdResult as SurdResult,
    run_mrmr as run_mrmr,
    run_mrmr_from_dict as run_mrmr_from_dict,
    version as version,
)
from typing import List

__version__: str
__all__: List[str]

def run_mrmr_polars(
    df: "object",
    target: str,
    max_features: int = 10,
) -> List[FeatureRanking]: ...
