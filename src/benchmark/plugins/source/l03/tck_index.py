"""L03 20K-expansion source index: DMN TCK tables + test scenarios.

Builds ``external_knowledge/indexes/dmn-tck/combined/`` from the pinned TCK
snapshot (``source_registry.fetchers.l03_dmn_tck``):

- ``tables.jsonl``   — one parsed decision table per line (same row shape as
                       the Camunda combined index; FEEL-subset profile check
                       per rule entry, exactly like the legacy indexer)
- ``tck_tests.jsonl`` — one TCK test scenario per line, matched to a table
                       decision by resultNode name, with the TCK expected
                       value and the ``aob-dmn-static`` recomputed value
                       (``conformant`` flag: engine agrees with the TCK
                       expectation — the dual-oracle conformance subset)
- ``chains.jsonl``   — DRG chains (>=2 usable decisions, acyclic), same shape
                       as the Camunda chains index
- ``rejects.jsonl``  — honest per-file/per-table/per-scenario exclusions
                       (literal-expression decisions, FEEL/date/structured
                       values outside the static profile, engine-vs-TCK
                       disagreements)
- ``index.json``     — counts + hit-policy distribution

Run: ``.venv/bin/python -m benchmark.plugins.source.l03.tck_index``
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from lxml import etree

from benchmark.plugins.verifier.l03.dmn_model import (
    DMN_SUFFIXES,
    DmnParseError,
    DmnUnsupportedError,
    parse_definitions,
    table_digest,
)
from benchmark.plugins.verifier.l03.dmn_eval import evaluate_table
from benchmark.plugins.verifier.l03.feel import (
    FeelUnsupportedError,
    eval_output_entry,
    match_unary,
)
from source_registry.fetchers import l03_dmn_tck as fetcher

SOURCE_ID = fetcher.SOURCE_ID
INDEX_DIR = fetcher.INDEX_BASE / "combined"
XSI_TYPE = "{http://www.w3.org/2001/XMLSchema-instance}type"

_SIMPLE_XSD = {
    "xsd:decimal", "xsd:double", "xsd:float", "xsd:int", "xsd:integer",
    "xsd:long", "xsd:string", "xsd:boolean",
}


def _local(el: etree._Element) -> str:
    return etree.QName(el).localname


def _children(el: etree._Element, name: str) -> list[etree._Element]:
    return [c for c in el if isinstance(c.tag, str) and _local(c) == name]


def _parse_scalar(value_el: etree._Element) -> tuple[bool, Any, str]:
    """(ok, python_value, reason) for a TCK <value> element."""
    xtype = value_el.get(XSI_TYPE)
    text = (value_el.text or "").strip()
    if xtype in ("xsd:decimal", "xsd:double", "xsd:float", "xsd:int",
                 "xsd:integer", "xsd:long"):
        try:
            num: Any = int(text) if text.lstrip("-").isdigit() else float(text)
        except ValueError:
            return False, None, "tck_numeric_unparseable"
        return True, num, ""
    if xtype == "xsd:string":
        return True, text, ""
    if xtype == "xsd:boolean":
        return True, text == "true", ""
    if xtype in ("xsd:date", "xsd:time", "xsd:dateTime", "xsd:duration"):
        return False, None, "tck_temporal_value_unsupported"
    if xtype is not None:
        return False, None, "tck_value_type_%s" % xtype.replace(":", "_")
    # No xsi:type: number / boolean / plain string.
    if text in ("true", "false"):
        return True, text == "true", ""
    neg = text.lstrip("-")
    if neg and (neg.isdigit() or (neg.count(".") == 1 and neg.replace(".", "").isdigit())):
        return True, int(text) if neg.isdigit() else float(text), ""
    return True, text, ""


def _parse_expected(expected_el: etree._Element) -> tuple[bool, Any, str]:
    """(ok, value, reason) for a resultNode <expected> block."""
    values = _children(expected_el, "value")
    components = _children(expected_el, "component")
    lists = _children(expected_el, "list")
    if lists:
        out: list[Any] = []
        for item in _children(lists[0], "item"):
            ok, val, reason = _parse_expected(item)
            if not ok:
                return False, None, reason
            out.append(val)
        return True, out, ""
    if components:
        out: dict[str, Any] = {}
        for comp in components:
            name = comp.get("name") or ""
            inner = _children(comp, "expected")
            inner_values = _children(comp, "value")
            if not name:
                return False, None, "tck_structured_value_unsupported"
            if inner:
                ok, val, reason = _parse_expected(inner[0])
            elif len(inner_values) == 1:
                ok, val, reason = _parse_scalar(inner_values[0])
            else:
                ok, val, reason = False, None, "tck_structured_value_unsupported"
            if not ok:
                return False, None, reason
            out[name] = val
        return True, out, ""
    if len(values) == 1:
        return _parse_scalar(values[0])
    if not values:
        return False, None, "tck_expected_empty"
    return False, None, "tck_multi_value_unsupported"


def _table_profile_check(table: Any) -> tuple[bool, str]:
    for rule in table.rules:
        for clause, entry in zip(table.inputs, rule.input_entries):
            try:
                match_unary(entry, None, clause.type_ref)
            except FeelUnsupportedError as exc:
                return False, "inputEntry %r: %s" % (entry, str(exc)[:120])
        for entry in rule.output_entries:
            try:
                eval_output_entry(entry, {})
            except FeelUnsupportedError as exc:
                return False, "outputEntry %r: %s" % (entry, str(exc)[:200])
    return True, ""


def _row_for(unit_id: str, rel: str, dec_id: str, dec) -> dict[str, Any]:
    return {
        "unit_id": unit_id,
        "release_id": fetcher.TCK_RELEASE_ID,
        "constituent_role": "tck-testcases",
        "path": rel,
        "decision_id": dec_id,
        "decision_name": dec.name,
        "table_digest": table_digest(dec.table),
        "hit_policy": dec.table.hit_policy,
        "aggregation": dec.table.aggregation,
        "inputs": [
            {"expression": c.expression, "type_ref": c.type_ref, "label": c.label}
            for c in dec.table.inputs
        ],
        "outputs": [
            {"name": o.name, "type_ref": o.type_ref, "values": list(o.values)}
            for o in dec.table.outputs
        ],
        "rules": [
            {"id": r.id, "input_entries": list(r.input_entries),
             "output_entries": list(r.output_entries), "description": r.description}
            for r in dec.table.rules
        ],
        "required_decision_ids": list(dec.required_decision_ids),
    }


def _toposort(edges: dict[str, list[str]]) -> list[str] | None:
    order: list[str] = []
    permanent: set[str] = set()
    temporary: set[str] = set()

    def visit(node: str) -> bool:
        if node in permanent:
            return True
        if node in temporary:
            return False
        temporary.add(node)
        for dep in edges.get(node, []):
            if not visit(dep):
                return False
        temporary.discard(node)
        permanent.add(node)
        order.append(node)
        return True

    for node in sorted(edges):
        if not visit(node):
            return None
    return order


def build_index(out_dir: Path = INDEX_DIR) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    root = fetcher.pinned_root()
    tables: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    # model rel path -> {decision_name: table row}
    models: dict[str, dict[str, dict[str, Any]]] = {}
    for path in sorted(root.rglob("*")):
        if not (path.is_file() and path.name.endswith(DMN_SUFFIXES)):
            continue
        rel = str(path.relative_to(root))
        unit_id = "%s/%s/%s" % (SOURCE_ID, fetcher.TCK_RELEASE_ID, rel)
        try:
            defs = parse_definitions(path)
        except DmnParseError as exc:
            rejects.append({"unit": unit_id, "reason": "parse_error", "detail": str(exc)[:200]})
            continue
        except DmnUnsupportedError as exc:
            rejects.append({"unit": unit_id, "reason": "profile_unsupported", "detail": str(exc)[:200]})
            continue
        file_decisions: dict[str, dict[str, Any]] = {}
        for dec_id, dec in defs.decisions.items():
            if dec.table is None:
                continue
            ok, reason = _table_profile_check(dec.table)
            if not ok:
                rejects.append({
                    "unit": unit_id, "decision": dec_id,
                    "reason": "profile_unsupported", "detail": reason,
                })
                continue
            row = _row_for(unit_id, rel, dec_id, dec)
            tables.append(row)
            file_decisions[dec_id] = row
        if file_decisions:
            models[rel] = {r["decision_name"]: r for r in file_decisions.values()}
            models[rel].update(file_decisions)
    # -- TCK test scenarios ----------------------------------------------------
    from benchmark.plugins.verifier.l03.dmn_model import table_from_payload

    def _payload(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "hit_policy": row["hit_policy"], "aggregation": row["aggregation"],
            "inputs": row["inputs"], "outputs": row["outputs"],
            "rules": row["rules"],
        }

    scenario_total = 0
    for path in sorted(root.rglob("*")):
        if not (path.is_file() and "-test" in path.stem and path.suffix == ".xml"):
            continue
        rel = str(path.relative_to(root))
        unit_id = "%s/%s/%s" % (SOURCE_ID, fetcher.TCK_RELEASE_ID, rel)
        try:
            tree = etree.parse(str(path))
        except etree.XMLSyntaxError as exc:
            rejects.append({"unit": unit_id, "reason": "parse_error", "detail": str(exc)[:200]})
            continue
        model_name = None
        for el in tree.iter():
            if isinstance(el.tag, str) and _local(el) == "modelName":
                model_name = (el.text or "").strip()
                break
        model_rel = _find_model(root, rel, model_name)
        if model_rel is None or model_rel not in models:
            rejects.append({
                "unit": unit_id, "reason": "tck_model_unusable",
                "detail": "model %r has no in-profile table decisions" % model_name,
            })
            continue
        by_name = models[model_rel]
        for tc in tree.iter():
            if not (isinstance(tc.tag, str) and _local(tc) == "testCase"):
                continue
            tc_id = tc.get("id") or ""
            inputs: dict[str, Any] = {}
            input_ok = True
            input_reason = ""
            for node in _children(tc, "inputNode"):
                name = node.get("name") or ""
                vals = _children(node, "value")
                if not name or len(vals) != 1:
                    input_ok, input_reason = False, "tck_input_shape"
                    break
                ok, val, reason = _parse_scalar(vals[0])
                if not ok:
                    input_ok, input_reason = False, reason
                    break
                inputs[name] = val
            if not input_ok:
                rejects.append({
                    "unit": unit_id, "test_case": tc_id,
                    "reason": "profile_unsupported", "detail": input_reason,
                })
                continue
            for node in _children(tc, "resultNode"):
                dec_name = node.get("name") or ""
                row = by_name.get(dec_name)
                if row is None:
                    continue
                expected_els = _children(node, "expected")
                if len(expected_els) != 1:
                    rejects.append({
                        "unit": unit_id, "test_case": tc_id, "decision": dec_name,
                        "reason": "profile_unsupported", "detail": "tck_expected_shape",
                    })
                    continue
                ok, expected, reason = _parse_expected(expected_els[0])
                if not ok:
                    rejects.append({
                        "unit": unit_id, "test_case": tc_id, "decision": dec_name,
                        "reason": "profile_unsupported", "detail": reason,
                    })
                    continue
                scenario_total += 1
                engine_value = None
                engine_error = ""
                try:
                    res = evaluate_table(table_from_payload(_payload(row)), inputs)
                    engine_value = res.value
                except DmnUnsupportedError as exc:
                    engine_error = str(exc)[:120]
                conformant = not engine_error and _values_equal(engine_value, expected)
                tests.append({
                    "unit_id": unit_id,
                    "model_path": model_rel,
                    "decision_id": row["decision_id"],
                    "decision_name": dec_name,
                    "test_case_id": tc_id,
                    "inputs": inputs,
                    "expected": expected,
                    "engine_value": engine_value,
                    "engine_error": engine_error,
                    "conformant": bool(conformant),
                })
    # -- chains ------------------------------------------------------------------
    by_unit: dict[str, list[dict[str, Any]]] = {}
    for row in tables:
        by_unit.setdefault(row["unit_id"], []).append(row)
    chains: list[dict[str, Any]] = []
    for unit_id, rows in sorted(by_unit.items()):
        if len(rows) < 2:
            continue
        ids = {r["decision_id"] for r in rows}
        edges = {
            r["decision_id"]: [d for d in r["required_decision_ids"] if d in ids]
            for r in rows
        }
        if not any(edges.values()):
            continue
        order = _toposort(edges)
        if order is None:
            rejects.append({
                "unit": unit_id, "reason": "profile_unsupported",
                "detail": "cyclic required-decision graph",
            })
            continue
        terminals = [d for d in order if not any(d in deps for deps in edges.values())]
        chains.append({
            "unit_id": unit_id,
            "release_id": rows[0]["release_id"],
            "constituent_role": rows[0]["constituent_role"],
            "path": rows[0]["path"],
            "decision_ids": order,
            "terminal_decision_ids": terminals,
            "edges": edges,
        })
    # -- write -------------------------------------------------------------------
    with (out_dir / "tables.jsonl").open("w", encoding="utf-8") as fh:
        for row in sorted(tables, key=lambda r: (r["unit_id"], r["decision_id"])):
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    with (out_dir / "tck_tests.jsonl").open("w", encoding="utf-8") as fh:
        for row in tests:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    with (out_dir / "rejects.jsonl").open("w", encoding="utf-8") as fh:
        for row in rejects:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    with (out_dir / "chains.jsonl").open("w", encoding="utf-8") as fh:
        for chain in chains:
            fh.write(json.dumps(chain, sort_keys=True) + "\n")
    hit_dist: dict[str, int] = {}
    for row in tables:
        key = row["hit_policy"] + ("+" + row["aggregation"] if row["aggregation"] else "")
        hit_dist[key] = hit_dist.get(key, 0) + 1
    conformant_rows = [t for t in tests if t["conformant"]]
    manifest = {
        "source_id": SOURCE_ID,
        "release_ids": [fetcher.TCK_RELEASE_ID],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine": fetcher.NATIVE_ENGINE,
        "tables_usable": len(tables),
        "tables_tck": len(tables),
        "chains_usable": len(chains),
        "tck_scenarios_total": scenario_total,
        "tck_scenarios_conformant": len(conformant_rows),
        "tck_scenarios_disagreed": scenario_total - len(conformant_rows),
        "rejects": len(rejects),
        "hit_policy_distribution": hit_dist,
        "conformant_decisions": len({t["decision_id"] for t in conformant_rows}),
        "wall_seconds": round(time.time() - t0, 2),
    }
    (out_dir / "index.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def _find_model(root: Path, test_rel: str, model_name: str | None) -> str | None:
    if model_name:
        direct = (Path(test_rel).parent / model_name)
        if (root / direct).is_file():
            return str(direct)
    for cand in sorted(Path(test_rel).parent.glob("*.dmn")):
        return str(cand)
    return None


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_values_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_values_equal(a[k], b[k]) for k in a)
    return a == b


def load_index(out_dir: Path = INDEX_DIR) -> dict[str, Any]:
    out_dir = Path(out_dir)
    with (out_dir / "tables.jsonl").open(encoding="utf-8") as fh:
        tables = [json.loads(line) for line in fh if line.strip()]
    with (out_dir / "tck_tests.jsonl").open(encoding="utf-8") as fh:
        tests = [json.loads(line) for line in fh if line.strip()]
    with (out_dir / "chains.jsonl").open(encoding="utf-8") as fh:
        chains = [json.loads(line) for line in fh if line.strip()]
    return {
        "manifest": json.loads((out_dir / "index.json").read_text()),
        "tables": tables,
        "tck_tests": tests,
        "chains": chains,
    }


def main() -> int:
    manifest = build_index()
    print(json.dumps(manifest, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
