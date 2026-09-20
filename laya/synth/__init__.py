"""Synthetic typed-question datasets from existing labelled image sets. See ``laya.synth.core``."""
from .core import DatasetWriter, Example, Pool, choice_q, noul_q, run_source, score_q, split_of
from .sources import PHOTO_SOURCES, SCREEN_SOURCES, SOURCES

__all__ = ["DatasetWriter", "Example", "Pool", "choice_q", "noul_q", "score_q", "split_of", "run_source",
           "SOURCES", "SCREEN_SOURCES", "PHOTO_SOURCES", "prepare"]


def prepare(root: str, source: str, n_train: int = 20000, n_val: int = 1000, seed: int = 0, max_side: int = 1024,
            val_frac: float = 0.05, log=print) -> dict:
    """Write ``<root>/synth_<source>/`` from the Hub. Returns the ``meta.json`` contents."""
    src = SOURCES[source](seed=seed)
    writer = DatasetWriter(root, "synth_" + source, max_side=max_side)
    try:
        return run_source(src, writer, n_train, n_val, seed=seed, val_frac=val_frac, log=log)
    except BaseException:
        writer.abort()
        raise
