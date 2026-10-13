"""Parser registry: task name -> parser function + version."""

from .gsm8k_answer import GSM8K_PARSER_VERSION, parse_gsm8k_answer
from .hotpotqa_answer import HOTPOTQA_PARSER_VERSION, parse_hotpotqa_answer
from .math_answer import MATH_PARSER_VERSION, parse_math_answer
from .musique_answer import MUSIQUE_PARSER_VERSION, parse_musique_answer
from .processbench_answer import PROCESSBENCH_PARSER_VERSION, parse_processbench_answer
from .reasoning import REASONING_PARSER_VERSION

# tasks.yaml parser names -> (function, version). The relaxed GSM8K parser is a
# diagnostic computed inside parse_gsm8k_answer; it is registered so reports
# can name it, but it is never a primary parser.
PARSERS = {
    "musique_answer": parse_musique_answer,
    "processbench_boxed_index": parse_processbench_answer,
    "gsm8k_official_regex": parse_gsm8k_answer,
    "math_boxed": parse_math_answer,
    "hotpotqa_free_text": parse_hotpotqa_answer,
}

PARSER_VERSIONS = {
    "musique_answer": MUSIQUE_PARSER_VERSION,
    "processbench_boxed_index": PROCESSBENCH_PARSER_VERSION,
    "gsm8k_official_regex": GSM8K_PARSER_VERSION,
    "gsm8k_relaxed": GSM8K_PARSER_VERSION,  # diagnostic only
    "math_boxed": MATH_PARSER_VERSION,
    "hotpotqa_free_text": HOTPOTQA_PARSER_VERSION,
    "reasoning_split": REASONING_PARSER_VERSION,
}

__all__ = ["PARSERS", "PARSER_VERSIONS"]
