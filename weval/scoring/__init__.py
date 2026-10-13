"""Scorer registry: task name -> scorer function + version."""

from .gsm8k_score import GSM8K_SCORER_VERSION, score_gsm8k
from .hotpotqa_score import HOTPOTQA_SCORER_VERSION, score_hotpotqa
from .math_score import MATH_SCORER_VERSION, score_math
from .musique_em_f1 import MUSIQUE_SCORER_VERSION, score_musique
from .processbench_score import PROCESSBENCH_SCORER_VERSION, score_processbench

SCORERS = {
    "musique_em_f1": score_musique,
    "processbench_official": score_processbench,
    "gsm8k_official": score_gsm8k,
    "math_equivalence": score_math,
    "hotpotqa_answer_f1": score_hotpotqa,
}

SCORER_VERSIONS = {
    "musique_em_f1": MUSIQUE_SCORER_VERSION,
    "processbench_official": PROCESSBENCH_SCORER_VERSION,
    "gsm8k_official": GSM8K_SCORER_VERSION,
    "math_equivalence": MATH_SCORER_VERSION,
    "hotpotqa_answer_f1": HOTPOTQA_SCORER_VERSION,
}

__all__ = ["SCORERS", "SCORER_VERSIONS"]
