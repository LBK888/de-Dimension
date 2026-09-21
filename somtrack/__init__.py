"""SOMTrack -- locomotion metrics, SOM clustering and publication figures.

A Python refactor of ``SOM tracking_v1.2b.ijm`` (Wang, Ho & Liao, NTOU 2020).

Typical use::

    from somtrack import AnalysisConfig, io_tables, pipeline

    cfg = AnalysisConfig()
    spots = io_tables.build_spot_dataset(files, colmap, group_ids, replicate_ids)
    feats = pipeline.features_from_spots(spots, cfg)
    res   = pipeline.run_analysis(feats, cfg)
    pipeline.export_all(res)
"""

from .citations import CITATIONS, Citation, MethodsLog
from .config import (AnalysisConfig, EmbeddingConfig, ExportConfig,
                     FigureConfig, NodeClusterConfig, PreprocessConfig,
                     ReportConfig, SomConfig, StatsConfig, TrackConfig)

__version__ = "3.0.0"
__all__ = [
    "AnalysisConfig", "TrackConfig", "PreprocessConfig", "SomConfig",
    "EmbeddingConfig", "StatsConfig", "NodeClusterConfig", "FigureConfig",
    "ExportConfig", "ReportConfig",
    "Citation", "CITATIONS", "MethodsLog",
    "__version__",
]
