"""Decision-table evaluation with DRG chaining for ``aob-dmn-static``.

``evaluate_decision`` evaluates one decision by id: required decisions are
evaluated first (depth-first, memoized per call), their single-output values
bound under the output clause name; table inputs then resolve from the merged
scope. Hit policies follow the DMN 1.1 semantics that Camunda implements for
the supported profile: UNIQUE (exactly one match), ANY (all matches agree),
FIRST (first match in rule order), PRIORITY / OUTPUT ORDER (ordered by the
output clause's declared values, falling back to rule order when no values
are declared), RULE ORDER and COLLECT (lists in rule order) with the
SUM/MIN/MAX/COUNT aggregations for single-output COLLECT tables.

Every evaluation returns a :class:`DecisionEval` carrying the matched rule
ids and any policy violations, so the native verifier can record exact,
recomputable evidence (engine id/version included by the caller).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmark.plugins.verifier.l03.dmn_model import (
    Decision,
    DecisionTable,
    Definitions,
    DmnUnsupportedError,
)
from benchmark.plugins.verifier.l03.feel import eval_output_entry, match_unary


class MissingInputError(DmnUnsupportedError):
    """A required input variable is unbound in the supplied scope."""


@dataclass(frozen=True)
class DecisionEval:
    decision_id: str
    value: Any
    matched_rule_ids: tuple[str, ...]
    matched_entries: tuple[tuple[Any, ...], ...]
    violations: tuple[str, ...] = ()


def matched_rules(table: DecisionTable, scope: dict[str, Any]) -> tuple[int, ...]:
    """Indexes of rules whose input entries all match the scope."""
    hits: list[int] = []
    for idx, rule in enumerate(table.rules):
        ok = True
        for clause, test in zip(table.inputs, rule.input_entries):
            if clause.expression not in scope:
                raise MissingInputError(
                    "input variable %r unbound (table inputs: %s)"
                    % (clause.expression, [c.expression for c in table.inputs])
                )
            value = scope[clause.expression]
            if isinstance(value, (list, tuple)):
                # COLLECT-chained value: membership semantics per element.
                if not any(match_unary(test, v, clause.type_ref) for v in value):
                    ok = False
                    break
            elif not match_unary(test, value, clause.type_ref):
                ok = False
                break
        if ok:
            hits.append(idx)
    return tuple(hits)


def rule_output(table: DecisionTable, rule_idx: int, scope: dict[str, Any]) -> tuple[Any, ...]:
    rule = table.rules[rule_idx]
    return tuple(eval_output_entry(entry, scope) for entry in rule.output_entries)


def apply_hit_policy(table: DecisionTable, hits: tuple[int, ...],
                     scope: dict[str, Any]) -> DecisionEval:
    entries = tuple(rule.output_entries for rule in (table.rules[i] for i in hits))
    evaluated = tuple(rule_output(table, i, scope) for i in hits)
    rule_ids = tuple(table.rules[i].id for i in hits)
    violations: list[str] = []
    policy = table.hit_policy
    single = len(table.outputs) == 1

    def _col(values_row: tuple[Any, ...]) -> Any:
        return values_row[0] if single else dict(
            zip([o.name for o in table.outputs], values_row)
        )

    if policy == "UNIQUE":
        if len(evaluated) > 1:
            violations.append("UNIQUE policy: %d rules matched" % len(evaluated))
        value = _col(evaluated[0]) if len(evaluated) == 1 else None
    elif policy == "ANY":
        if len({e for e in evaluated}) > 1:
            violations.append("ANY policy: matched rules disagree")
        value = _col(evaluated[0]) if evaluated and len({e for e in evaluated}) == 1 else None
    elif policy == "FIRST":
        value = _col(evaluated[0]) if evaluated else None
    elif policy == "RULE ORDER":
        value = [_col(e) for e in evaluated]
    elif policy in ("PRIORITY", "OUTPUT ORDER"):
        ordered = _priority_order(table, hits)
        if policy == "PRIORITY":
            value = _col(rule_output(table, ordered[0], scope)) if ordered else None
            rule_ids = tuple(table.rules[i].id for i in ordered)
        else:
            value = [_col(rule_output(table, i, scope)) for i in ordered]
            rule_ids = tuple(table.rules[i].id for i in ordered)
    elif policy == "COLLECT":
        collected = [_col(e) for e in evaluated]
        agg = table.aggregation
        if not agg:
            value = collected
        elif agg == "COUNT":
            value = len(collected)
        else:
            flat = [v for row in collected for v in (row.values() if isinstance(row, dict) else (row,))]
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in flat):
                raise DmnUnsupportedError("COLLECT %s over non-numeric outputs" % agg)
            if agg == "SUM":
                value = sum(flat)
            elif agg == "MIN":
                value = min(flat) if flat else None
            elif agg == "MAX":
                value = max(flat) if flat else None
            else:  # pragma: no cover - guarded by the parser
                raise DmnUnsupportedError("aggregation %r unsupported" % agg)
    else:  # pragma: no cover - guarded by the parser
        raise DmnUnsupportedError("hit policy %r unsupported" % policy)
    return DecisionEval(
        decision_id="", value=value, matched_rule_ids=rule_ids,
        matched_entries=entries, violations=tuple(violations),
    )


def _priority_order(table: DecisionTable, hits: tuple[int, ...]) -> tuple[int, ...]:
    """Order matched rule indexes by the first output clause's declared values.

    Falls back to rule order for tables without declared output values.
    """
    values = table.outputs[0].values
    if not values:
        return hits
    rank = {v: i for i, v in enumerate(values)}

    def key(rule_idx: int) -> tuple[int, int]:
        raw = table.rules[rule_idx].output_entries[0].strip().strip('"')
        return (rank.get(raw, len(rank)), rule_idx)

    return tuple(sorted(hits, key=key))


def evaluate_table(table: DecisionTable, scope: dict[str, Any],
                   decision_id: str = "") -> DecisionEval:
    hits = matched_rules(table, scope)
    result = apply_hit_policy(table, hits, scope)
    return DecisionEval(
        decision_id=decision_id, value=result.value,
        matched_rule_ids=result.matched_rule_ids,
        matched_entries=result.matched_entries, violations=result.violations,
    )


def evaluate_chain(
    tables: dict[str, DecisionTable],
    edges: dict[str, list[str]],
    terminal: str,
    inputs: dict[str, Any],
    decision_names: dict[str, str] | None = None,
) -> tuple[DecisionEval, dict[str, DecisionEval]]:
    """Evaluate a DRG chain of payload-built tables (dependency-first).

    ``edges`` maps decision id -> required decision ids. Each decision's output
    is bound under its first output clause name (fallback: decision name), so
    consuming tables reference it through their input expressions.
    """
    order = _chain_order(edges)
    if terminal not in tables:
        raise DmnUnsupportedError("terminal decision %r not in tables" % terminal)
    scope = dict(inputs)
    memo: dict[str, DecisionEval] = {}
    for dec_id in order:
        if dec_id not in tables:
            continue
        result = evaluate_table(tables[dec_id], scope, decision_id=dec_id)
        memo[dec_id] = result
        table = tables[dec_id]
        if table.outputs:
            scope[table.outputs[0].name] = result.value
        elif decision_names and decision_names.get(dec_id):
            scope[decision_names[dec_id]] = result.value
        if isinstance(result.value, dict):
            for key, val in result.value.items():
                scope[key] = val
    return memo[terminal], memo


def _chain_order(edges: dict[str, list[str]]) -> list[str]:
    order: list[str] = []
    permanent: set[str] = set()
    temporary: set[str] = set()

    def visit(node: str) -> None:
        if node in permanent:
            return
        if node in temporary:
            raise DmnUnsupportedError("cyclic required-decision graph")
        temporary.add(node)
        for dep in edges.get(node, []):
            visit(dep)
        temporary.discard(node)
        permanent.add(node)
        order.append(node)

    for node in sorted(edges):
        visit(node)
    return order
    """Evaluate ``decision_id`` with DRG chaining; returns (result, sub-evals)."""
    memo: dict[str, DecisionEval] = {}

    def _eval(dec_id: str, scope: dict[str, Any]) -> DecisionEval:
        if dec_id in memo:
            return memo[dec_id]
        dec: Decision = defs.decisions.get(dec_id)  # type: ignore[assignment]
        if dec is None:
            raise DmnUnsupportedError("required decision %r not in definitions" % dec_id)
        merged = dict(scope)
        for req_id in dec.required_decision_ids:
            sub = _eval(req_id, scope)
            table = defs.decision_table(req_id)
            out_name = table.outputs[0].name if table.outputs else defs.decisions[req_id].name
            merged[out_name] = sub.value
            if len(table.outputs) > 1 and isinstance(sub.value, dict):
                for k, v in sub.value.items():
                    merged[k] = v
        table = defs.decision_table(dec_id)
        result = evaluate_table(table, merged, decision_id=dec_id)
        memo[dec_id] = result
        return result

    final = _eval(decision_id, dict(inputs))
    return final, memo
