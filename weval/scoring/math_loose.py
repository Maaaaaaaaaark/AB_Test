"""Looser MATH answer equivalence - a DIAGNOSTIC, never the primary score.

The official MATH checker (vendored ``is_equiv``) compares normalized
strings, so mathematically equal answers written differently can be scored
wrong (``0.5`` vs ``\\frac{1}{2}``, ``x=3`` vs ``3``, ``5\\text{ cm}`` vs
``5``, ``2\\pi`` vs ``\\pi\\cdot 2``). Models differ in how they write
answers, which can bias a model comparison. This module gives a second,
looser opinion so that disagreements can be listed and spot-checked; the
reported MATH accuracy is ALWAYS the official one.

Checks, in order (True as soon as one succeeds):
  1. equal after a wider normalization (wrappers, spacing, \\dfrac, units in
     \\text{}, degrees, a mathematically scaled trailing percent, a leading
     ``x =``, trailing period);
  2. the official ``is_equiv`` on the normalized strings;
  3. numeric equality (rel. tol 1e-6) of expressions built from numbers,
     \\frac, \\sqrt, \\pi, + - * / ^ and parentheses, evaluated with a
     restricted AST evaluator (no names other than pi / sqrt, no calls other
     than sqrt);
  4. the same element-wise for tuples / lists / intervals with the same
     brackets and the same number of elements;
  5. a single multiple-choice letter, e.g. ``\\text{(C)}`` vs ``C``.
Two missing answers are never equal.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import Callable, List, Optional

MATH_LOOSE_VERSION = "math-loose-v1"

_SPACING = re.compile(r"\\(?:displaystyle|left|right|!|,|;|:|quad|qquad|limits)(?![a-zA-Z])")
_TEXT_WRAP = re.compile(r"\\(?:text|textbf|textit|mathrm|mathbf|mbox|operatorname)\s*\{([^{}]*)\}")
_DEGREES = re.compile(r"\^\s*\{?\\circ\}?|°|\\degree")
_LEADING_VAR = re.compile(r"^[a-zA-Z](?:_\{?[a-zA-Z0-9]+\}?)?\s*=\s*(?=\S)")
_UNIT_WORDS = re.compile(
    r"(?<=[0-9}])\s*(?:\\text\{\s*[a-zA-Z ]+\s*\}|(?:cm|mm|km|m|inches|inch|feet|ft|units?|"
    r"degrees?|dollars?|cents?|sq|square|meters?|seconds?|minutes?|hours?|days?)\b(?:\^2|\^3)?)\s*$"
)


def _normalize(s: str) -> str:
    s = s.strip()
    s = s.strip("$").strip()
    s = s.replace("\\(", "").replace("\\)", "").replace("\\[", "").replace("\\]", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = _DEGREES.sub("", s)
    # A terminal percent is a scale, not decoration: 50% == 0.5, not 50.
    # Restrict this to a trailing marker so unsupported prose containing a
    # percent sign remains unsupported rather than being silently changed.
    is_percent = re.search(r"(?:\\%|%)\s*$", s) is not None
    if is_percent:
        s = re.sub(r"(?:\\%|%)\s*$", "", s)
    s = _UNIT_WORDS.sub("", s)
    s = _SPACING.sub("", s)
    prev = None
    while prev != s:  # unwrap nested \text{...}
        prev = s
        s = _TEXT_WRAP.sub(r"\1", s)
    s = re.sub(r"([\^_])\{([^{}])\}", r"\1\2", s)  # x^{2} -> x^2, a_{1} -> a_1
    s = _LEADING_VAR.sub("", s.strip())
    s = s.rstrip(".").strip()
    s = s.replace(" ", "")
    if is_percent and s:
        s = f"({s})/100"
    return s


def _brace_arg(s: str, i: int) -> tuple[str, int]:
    """Content of the {...} group starting at s[i] == '{' and the index after it."""
    if i >= len(s) or s[i] != "{":
        # single-character argument, e.g. \frac12
        if i < len(s):
            return s[i], i + 1
        raise ValueError("missing argument")
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1 : j], j + 1
    raise ValueError("unbalanced braces")


def _latex_to_expr(s: str) -> str:
    out: List[str] = []
    i = 0
    while i < len(s):
        if s.startswith("\\frac", i):
            a, i = _brace_arg(s, i + 5)
            b, i = _brace_arg(s, i)
            out.append(f"(({_latex_to_expr(a)})/({_latex_to_expr(b)}))")
        elif s.startswith("\\sqrt", i):
            if i + 5 < len(s) and s[i + 5] == "[":
                raise ValueError("n-th roots not supported")
            a, i = _brace_arg(s, i + 5)
            out.append(f"sqrt({_latex_to_expr(a)})")
        elif s.startswith("\\pi", i):
            out.append("pi")
            i += 3
        elif s.startswith("\\cdot", i):
            out.append("*")
            i += 5
        elif s.startswith("\\times", i):
            out.append("*")
            i += 6
        elif s[i] == "\\":
            raise ValueError("unsupported command")
        elif s[i] == "{":
            out.append("(")
            i += 1
        elif s[i] == "}":
            out.append(")")
            i += 1
        elif s[i] == "^":
            out.append("**")
            i += 1
        else:
            out.append(s[i])
            i += 1
    expr = "".join(out)
    # implicit multiplication: 2pi, 3sqrt(2), 2(3), )(  , )2
    expr = re.sub(r"(?<=[0-9)])\s*(?=(?:pi|sqrt|\())", "*", expr)
    expr = re.sub(r"(?<=pi)\s*(?=[0-9(]|sqrt|pi)", "*", expr)
    return expr


_BINOPS: dict = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow,
}
_UNOPS: dict = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return float(_BINOPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNOPS:
        return float(_UNOPS[type(node.op)](_eval_node(node.operand)))
    if isinstance(node, ast.Name) and node.id == "pi":
        return math.pi
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sqrt"
            and len(node.args) == 1 and not node.keywords):
        return math.sqrt(_eval_node(node.args[0]))
    raise ValueError("disallowed expression")


def to_number(s: str) -> Optional[float]:
    """Numeric value of a simple LaTeX expression, or None."""
    if not s:
        return None
    t = s
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", t):  # thousands separators
        t = t.replace(",", "")
    if "," in t:
        return None
    try:
        expr = _latex_to_expr(t)
        if not expr or len(expr) > 200:
            return None
        val = _eval_node(ast.parse(expr, mode="eval"))
    except Exception:  # noqa: BLE001 - not a simple numeric expression
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def _num_equal(a: str, b: str) -> bool:
    x, y = to_number(a), to_number(b)
    if x is None or y is None:
        return False
    return math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-9)


def _split_top_level(s: str) -> List[str]:
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def _choice_letter(s: str) -> Optional[str]:
    m = re.fullmatch(r"\(?([A-E])\)?", s)
    return m.group(1) if m else None


def loose_equiv(pred: Optional[str], gold: Optional[str],
                official: Optional[Callable[[str, str], bool]] = None) -> bool:
    if pred is None or gold is None:
        return False
    p, g = _normalize(str(pred)), _normalize(str(gold))
    if not p or not g:
        return False
    if p == g:
        return True
    if official is not None:
        try:
            if official(p, g):
                return True
        except Exception:  # noqa: BLE001
            pass
    if _num_equal(p, g):
        return True
    # tuples / lists / intervals: same brackets, same arity, element-wise
    for strip in (("(", ")"), ("[", "]"), ("\\{", "\\}"), ("", "")):
        lo, hi = strip
        if p.startswith(lo) and p.endswith(hi) and g.startswith(lo) and g.endswith(hi):
            pi_ = p[len(lo): len(p) - len(hi)] if hi else p
            gi = g[len(lo): len(g) - len(hi)] if hi else g
            pp, gg = _split_top_level(pi_), _split_top_level(gi)
            if len(pp) > 1 and len(pp) == len(gg):
                if all(a == b or _num_equal(a, b) or (official is not None and _safe(official, a, b))
                       for a, b in zip(pp, gg)):
                    return True
            break
    cp, cg = _choice_letter(p), _choice_letter(g)
    if cp is not None and cp == cg:
        return True
    return False


def _safe(fn: Callable[[str, str], bool], a: str, b: str) -> bool:
    try:
        return bool(fn(a, b))
    except Exception:  # noqa: BLE001
        return False
