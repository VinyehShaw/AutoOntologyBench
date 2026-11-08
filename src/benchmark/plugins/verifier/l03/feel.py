"""FEEL-subset unary-test and output-entry evaluation for ``aob-dmn-static``.

Supported unary-test atoms (per top-level comma alternative):
- empty / ``-``                 -> always true
- ``"literal"``                 -> string equality
- bare word (``Electronic``)    -> string equality (Camunda legacy test style)
- ``123`` / ``1.5``             -> numeric equality
- ``true`` / ``false``          -> boolean equality
- ``< <= > >= = !=`` + literal  -> comparison (numeric or lexicographic)
- ``[a..b]`` / ``(a..b)`` / mixed brackets -> numeric interval membership
- ``cellInput == "x"``          -> legacy equality (engine-test dialect)
- ``not(<alternatives>)``       -> negation of the inner disjunction

Anything else raises :class:`FeelUnsupportedError`; the source inventory then
marks the owning table ``profile_unsupported`` (honest exclusion, no guessing).
Output entries support string/number/boolean literals, ``null``, and variable
references resolved against the current scope (with bare-word string fallback,
again the Camunda legacy style).
"""
from __future__ import annotations

import re
from typing import Any

from benchmark.plugins.verifier.l03.dmn_model import DmnUnsupportedError

__all__ = ["FeelUnsupportedError", "match_unary", "eval_output_entry", "split_alternatives"]


class FeelUnsupportedError(DmnUnsupportedError):
    """Unary test / output entry outside the supported FEEL subset."""


_CMP_RE = re.compile(r"^(<=?|>=?|!=?)\s*(.+)$", re.S)
_INTERVAL_RE = re.compile(
    r"^([\[\(\]])\s*(-?\d+(?:\.\d+)?)\s*\.\.\s*(-?\d+(?:\.\d+)?)\s*([\]\)\[])$"
)
_NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_LEGACY_EQ_RE = re.compile(r"^cellInput\s*==\s*(.+)$", re.S)
_BARE_WORD_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_ \-.]*$")
_STRING_RE = re.compile(r'^"(?:[^"\\]|\\.)*"$', re.S)


def split_alternatives(text: str) -> list[str]:
    """Split on top-level commas, respecting quotes, brackets and parens."""
    out: list[str] = []
    cur: list[str] = []
    quote: str | None = None
    depth = 0
    for ch in text:
        if quote is not None:
            cur.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            cur.append(ch)
            continue
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur).strip())
    return [p for p in out if p != ""]


def _unquote(text: str) -> str:
    return text[1:-1].replace('\\"', '"').replace("\\\\", "\\")


def _parse_number(text: str) -> int | float:
    return int(text) if re.fullmatch(r"-?\d+", text) else float(text)


def match_unary(test: str, value: Any, type_ref: str = "string") -> bool:
    """True when typed ``value`` satisfies unary-test ``test``."""
    text = test.strip()
    if text in ("", "-"):
        return True
    if text.startswith("not(") and text.endswith(")"):
        inner = text[4:-1]
        return not any(
            _atom_matches(a, value, type_ref) for a in split_alternatives(inner)
        )
    return any(_atom_matches(a, value, type_ref) for a in split_alternatives(text))


def _atom_matches(atom: str, value: Any, type_ref: str) -> bool:
    atom = atom.strip()
    if atom in ("", "-"):
        return True
    legacy = _LEGACY_EQ_RE.match(atom)
    if legacy:
        return _atom_matches(legacy.group(1).strip(), value, type_ref)
    if _STRING_RE.match(atom):
        return isinstance(value, str) and value == _unquote(atom)
    if atom in ("true", "false"):
        return value is (atom == "true")
    if _NUMBER_RE.match(atom):
        return _is_number(value) and float(value) == float(_parse_number(atom))
    interval = _INTERVAL_RE.match(atom)
    if interval:
        left, lo, hi, right = interval.groups()
        if not _is_number(value):
            return False
        v = float(value)
        lo_v, hi_v = float(lo), float(hi)
        lo_ok = v >= lo_v if left == "[" else v > lo_v
        hi_ok = v <= hi_v if right == "]" else v < hi_v
        return lo_ok and hi_ok
    cmp_m = _CMP_RE.match(atom)
    if cmp_m:
        return _compare(cmp_m.group(1), cmp_m.group(2).strip(), value)
    if _BARE_WORD_RE.match(atom):
        return isinstance(value, str) and value == atom
    raise FeelUnsupportedError("unary atom %r outside supported FEEL subset" % atom)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _compare(op: str, literal_text: str, value: Any) -> bool:
    literal: Any
    if _STRING_RE.match(literal_text):
        literal = _unquote(literal_text)
        if not isinstance(value, str):
            return False
    elif _NUMBER_RE.match(literal_text):
        literal = _parse_number(literal_text)
        if not _is_number(value):
            return False
        value, literal = float(value), float(literal)
    elif literal_text in ("true", "false"):
        literal = literal_text == "true"
        if not isinstance(value, bool):
            return False
    else:
        raise FeelUnsupportedError(
            "comparison literal %r outside supported FEEL subset" % literal_text
        )
    if op == "<":
        return value < literal
    if op == "<=":
        return value <= literal
    if op == ">":
        return value > literal
    if op == ">=":
        return value >= literal
    if op == "=":
        return value == literal
    if op == "!=":
        return value != literal
    raise FeelUnsupportedError("comparison operator %r unsupported" % op)


def eval_output_entry(text: str, scope: dict[str, Any]) -> Any:
    """Evaluate one output literal expression against the variable scope."""
    entry = text.strip()
    if entry in ("", "-"):
        return None
    if _STRING_RE.match(entry):
        return _unquote(entry)
    if _NUMBER_RE.match(entry):
        return _parse_number(entry)
    if entry == "true":
        return True
    if entry == "false":
        return False
    if entry == "null":
        return None
    if _BARE_WORD_RE.match(entry):
        if entry in scope:
            return scope[entry]
        return entry  # bare-word string constant (Camunda legacy style)
    raise FeelUnsupportedError("output entry %r outside supported FEEL subset" % entry)
