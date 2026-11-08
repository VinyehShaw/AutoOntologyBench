"""L07 source plugin: index the pinned BioModels snapshot via libsbml + solver screen.

``build_index`` walks the pinlock (the fetcher's integrity anchor), parses each
model with libsbml into a typed structure summary, then runs the lane's
pinned CVODE protocol twice per candidate model (bit-identical check) to
record per-species trajectory features. Output under
``external_knowledge/indexes/biomodels/<release_id>/``:

- ``models.jsonl``  — one record per model: struct + screening verdict
- ``manifest.json`` — versions, counts, timings

The screening block is a proposal-time cache only: the native verifier
recomputes everything from the pinned SBML bytes at qualification time.
"""
from __future__ import annotations

import json
import os
import signal
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

from benchmark.plugins.source.l07.sbml_struct import (
    LIBSBML_VERSION, ModelStruct, SbmlStructError, parse_sbml_bytes,
)
from benchmark.plugins.source.l07.simulate import (
    PROTOCOL, SimulationError, simulate_checked,
)
from benchmark.schema.plugin_protocols import IndexManifest, SourceSnapshot, SourceUnit
from source_registry.fetchers.l07_biomodels import (
    INDEX_ROOT, PINLOCK_PATH, PINNED_ROOT, RELEASE_ID, SOURCE_ID, BioModelsSource,
)

MAX_SIM_SPECIES = 40
SCREEN_ALARM_S = 60


class _Alarm:
    def __init__(self, seconds: int) -> None:
        self.seconds = seconds

    def __enter__(self) -> None:
        signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(self.seconds)

    def __exit__(self, *exc: Any) -> None:
        signal.alarm(0)


def _on_alarm(signum: int, frame: Any) -> None:
    raise SimulationTimeout("screening exceeded alarm")


class SimulationTimeout(RuntimeError):
    pass


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class L07SourcePlugin:
    source_id = SOURCE_ID

    def __init__(self, release_root: Path | None = None) -> None:
        self.release_root = Path(release_root) if release_root else PINNED_ROOT

    @property
    def pinlock(self) -> dict[str, Any]:
        return json.loads(PINLOCK_PATH.read_text(encoding="utf-8"))

    def sbml_path(self, model_id: str) -> Path:
        entry = self.pinlock["models"][model_id]
        return self.release_root / "models" / model_id / entry["filename"]

    def read_model(self, model_id: str) -> bytes:
        return self.sbml_path(model_id).read_bytes()

    def enumerate_units(self, snapshot: SourceSnapshot | None = None) -> Iterator[SourceUnit]:
        src = BioModelsSource()
        snap = snapshot or SourceSnapshot(
            source_id=SOURCE_ID, release_id=RELEASE_ID, root=self.release_root,
            descriptor=src.describe())
        yield from src.inventory(snap)

    # -- §3.2 SourcePlugin facade -------------------------------------------

    def pin_and_index(self, source_config: dict[str, Any] | None = None) -> SourceSnapshot:
        """Verify the pinned snapshot and (re)build the screening index if asked.

        The pinned BioModels subset is fetched/verified by
        ``source_registry.fetchers.l07_biomodels``; this adapter never mutates
        the pinlock. Indexing is skipped when a complete index is already on
        disk unless ``source_config["rebuild_index"]`` is true.
        """
        cfg = source_config or {}
        if not PINLOCK_PATH.is_file():
            raise FileNotFoundError(f"pinlock missing: {PINLOCK_PATH}")
        src = BioModelsSource()
        snapshot = SourceSnapshot(
            source_id=SOURCE_ID, release_id=RELEASE_ID, root=self.release_root,
            descriptor=src.describe())
        index_file = INDEX_ROOT / "models.jsonl"
        if cfg.get("rebuild_index") or not index_file.is_file():
            self.build_index(INDEX_ROOT)
        return snapshot

    # -- screening -----------------------------------------------------------

    def screen_model(self, model_id: str) -> dict[str, Any]:
        raw = self.read_model(model_id)
        rec: dict[str, Any] = {"model_id": model_id}
        try:
            struct = parse_sbml_bytes(raw)
        except SbmlStructError as exc:
            rec["screening"] = {"parse_ok": False, "error": str(exc)[:200]}
            return rec
        rec["struct"] = struct.to_json()
        floating = struct.floating_species
        sim_species = floating[:MAX_SIM_SPECIES]
        screening: dict[str, Any] = {
            "parse_ok": True,
            "n_species": len(struct.species),
            "n_floating_species": len(floating),
            "n_reactions": len(struct.reactions),
            "n_settable_parameters": len(struct.settable_parameters),
            "n_reactions_with_kinetic_law": sum(
                1 for r in struct.reactions if r.kinetic_formula),
        }
        if not floating:
            screening["sim_ok"] = False
            screening["error"] = "no floating species"
        else:
            try:
                with _Alarm(SCREEN_ALARM_S):
                    sim = simulate_checked(raw.decode("utf-8"), sim_species)
                screening["sim_ok"] = True
                screening["result_sha256"] = sim["result_sha256"]
                screening["protocol"] = dict(PROTOCOL)
                screening["species"] = sim["species"]
                screening["missing_species"] = sim["missing_species"]
            except (SimulationError, SimulationTimeout) as exc:
                screening["sim_ok"] = False
                screening["error"] = str(exc)[:200]
        rec["screening"] = screening
        return rec

    def build_index(self, out_dir: Path, *, log_path: Path | None = None) -> IndexManifest:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        model_ids = sorted(self.pinlock["models"])
        counts = {"models": 0, "parse_ok": 0, "sim_ok": 0, "failed": 0}
        log_fh = log_path.open("a", encoding="utf-8") if log_path else None
        try:
            with (out_dir / "models.jsonl").open("w", encoding="utf-8") as fh:
                for i, mid in enumerate(model_ids):
                    rec = self.screen_model(mid)
                    rec["release_id"] = RELEASE_ID
                    fh.write(json.dumps(rec, sort_keys=True) + "\n")
                    counts["models"] += 1
                    scr = rec.get("screening", {})
                    if screening_ok(scr):
                        counts["sim_ok"] += 1
                        counts["parse_ok"] += 1
                    elif scr.get("parse_ok"):
                        counts["parse_ok"] += 1
                    else:
                        counts["failed"] += 1
                    if log_fh and (i + 1) % 20 == 0:
                        log_fh.write(f"screened {i + 1}/{len(model_ids)} "
                                     f"sim_ok={counts['sim_ok']}\n")
                        log_fh.flush()
        finally:
            if log_fh:
                log_fh.close()
        manifest = {
            "source_id": SOURCE_ID, "release_id": RELEASE_ID,
            "libsbml_version": LIBSBML_VERSION,
            "protocol": dict(PROTOCOL),
            "counts": counts,
            "wall_seconds": round(time.time() - t0, 1),
        }
        _write_json_atomic(out_dir / "manifest.json", manifest)
        return IndexManifest(
            source_id=SOURCE_ID, release_id=RELEASE_ID, out_dir=str(out_dir),
            unit_count=counts["models"],
            kind_counts={"sbml_model": counts["models"],
                         "sim_runnable": counts["sim_ok"]})

    @staticmethod
    def load_index(out_dir: Path) -> dict[str, Any]:
        out_dir = Path(out_dir)
        models: dict[str, dict[str, Any]] = {}
        with (out_dir / "models.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                models[rec["model_id"]] = rec
        return {
            "manifest": json.loads((out_dir / "manifest.json").read_text()),
            "models": models,
        }


def screening_ok(scr: dict[str, Any]) -> bool:
    return bool(scr.get("parse_ok")) and bool(scr.get("sim_ok"))


def load_struct(rec: dict[str, Any]) -> ModelStruct:
    return ModelStruct.from_json(rec["struct"])


PLUGIN = L07SourcePlugin()
__all__ = ["L07SourcePlugin", "PLUGIN", "load_struct", "screening_ok"]
