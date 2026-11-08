"""Safe answer normalisation + process isolation for the L07 native verifier.

The L07 verifier recomputes every verdict with libSBML and libRoadRunner.
Those native libraries can abort the Python process (e.g. a SIGSEGV inside
``roadrunner.RoadRunner()`` for one pinned model), which would kill the whole
scorer run for an otherwise valid prediction.  Two guards live here:

* :func:`normalize_answer` converts unambiguous answer wrappers the model
  produced (a raw ``answer_ids``/``ranking`` list, or a ranking list of
  ``{"species_id": ...}`` objects) into the shape the lane verifier reads,
  raising :class:`AnswerShapeError` for shapes that cannot be interpreted
  safely.
* :func:`verify_isolated` runs ``verify_terminal`` in a forked child, so a
  native crash becomes a deterministic verdict envelope instead of a dead
  scorer.  The parent's loaded verifier/index is inherited copy-on-write, so
  the lane semantics are unchanged.
"""
from __future__ import annotations

import multiprocessing as mp
import signal
from typing import Any

from benchmark.eval_20k.lanes import _json_safe

#: kinds whose verifier reads ``{"answer_ids": [...]}``.
_SET_KINDS = frozenset({"retention_set", "pulse_set", "reaction_degree_query"})
#: kinds whose verifier reads ``{"ranking": [...]}``.
_RANKING_KINDS = frozenset({"peak_time_ranking", "ranking"})
#: kinds whose verifier reads ``{"accepted_values": [...]}``.
_ACCEPTED_KINDS = frozenset({"init_threshold_set", "init_window_set"})

_SPECIES_ID_KEYS = ("species_id", "species")
_REACTION_ID_KEYS = ("reaction_id", "reaction")

SHAPE_EXCEPTIONS = frozenset({"AttributeError", "TypeError", "KeyError"})


class AnswerShapeError(ValueError):
    """The submitted answer cannot be normalised to a verifier shape."""


def _item_id(item: Any, keys: tuple[str, ...]) -> str | None:
    if isinstance(item, str) and item:
        return item
    if not isinstance(item, dict):
        return None
    values = [item[k] for k in keys
              if isinstance(item.get(k), str) and item.get(k)]
    return values[0] if len(values) == 1 else None


def _normalize_items(items: list[Any], keys: tuple[str, ...],
                     label: str) -> list[str]:
    out: list[str] = []
    for item in items:
        value = _item_id(item, keys)
        if value is None:
            raise AnswerShapeError(
                "%s items must be ids or objects with exactly one of %s"
                % (label, "/".join(keys)))
        out.append(value)
    return out


def _payload_list(answer: Any, key: str) -> list[Any] | None:
    if isinstance(answer, dict) and isinstance(answer.get(key), list):
        return answer[key]
    if isinstance(answer, list):
        return answer
    return None


def normalize_answer(answer: Any, oracle_payload: dict[str, Any]
                     ) -> tuple[Any, dict[str, Any] | None]:
    """Return ``(normalized_answer, note)`` for a clear L07 answer shape.

    Raises :class:`AnswerShapeError` when a list is present but its items are
    not unambiguously ids; the caller records that as a scored failure rather
    than calling the native verifier.
    """
    ast = oracle_payload.get("ast")
    kind = ast.get("kind") if isinstance(ast, dict) else None
    if not isinstance(kind, str):
        return answer, None

    if kind in _SET_KINDS:
        key = "answer_ids"
        items = _payload_list(answer, key)
        keys = _REACTION_ID_KEYS if kind == "reaction_degree_query" else _SPECIES_ID_KEYS
    elif kind in _RANKING_KINDS:
        key = "ranking"
        items = _payload_list(answer, key)
        keys = _SPECIES_ID_KEYS
    elif kind in _ACCEPTED_KINDS:
        key = "accepted_values"
        items = _payload_list(answer, key)
        keys = ()
    else:
        return answer, None

    if items is None:
        return answer, None
    if isinstance(answer, dict) and key in answer and all(
            isinstance(item, str) for item in items):
        return answer, None
    if keys:
        normalized = _normalize_items(items, keys, kind)
    else:
        if not all(isinstance(item, (int, float)) and not isinstance(item, bool)
                   for item in items):
            raise AnswerShapeError(
                "%s items must be numbers" % kind)
        normalized = items
    note = {
        "wrapper_key": key,
        "kind": kind,
        "original_type": type(answer).__name__,
    }
    return {key: normalized}, note


def _worker(send: Any, verifier: Any, task: Any, answer: Any) -> None:
    try:
        verdict = verifier.verify_terminal(task, answer)
        detail = getattr(verdict, "detail", {}) or {}
        if not isinstance(detail, dict):
            detail = {"detail": str(detail)}
        send.send({
            "ok": True,
            "valid": bool(getattr(verdict, "valid", False)),
            "detail": _json_safe(detail),
        })
    except BaseException as exc:  # noqa: BLE001 - report any worker failure
        try:
            send.send({
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            })
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        try:
            send.close()
        except OSError:
            pass


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return "signal_%d" % signum


def verify_isolated(verifier: Any, task: Any, answer: Any, *,
                    timeout: float = 600.0) -> dict[str, Any]:
    """Run ``verify_terminal`` in a forked child and return an envelope.

    A normal result is ``{"ok": True, "valid": bool, "detail": {...}}``.  A
    raised exception is ``{"ok": False, "error_type": ..., "error": ...}``.  A
    native crash or timeout is ``{"ok": False, "crash": True, "reason": ...}``
    with a deterministic signal/timeout description.
    """
    try:
        context = mp.get_context("fork")
        recv, send = context.Pipe(duplex=False)
        process = context.Process(
            target=_worker, args=(send, verifier, task, answer), daemon=True)
    except Exception as exc:
        return {
            "ok": False, "crash": True,
            "reason": "native_isolation_failed: %s: %s"
                      % (type(exc).__name__, exc),
        }
    try:
        process.start()
    except Exception as exc:
        recv.close()
        send.close()
        return {
            "ok": False, "crash": True,
            "reason": "native_isolation_failed: %s: %s"
                      % (type(exc).__name__, exc),
        }
    send.close()
    envelope: dict[str, Any] | None = None
    timed_out = False
    try:
        if recv.poll(timeout):
            try:
                envelope = recv.recv()
            except EOFError:
                envelope = None
        else:
            timed_out = True
    finally:
        recv.close()
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)

    if envelope is not None:
        return envelope
    if timed_out:
        return {
            "ok": False, "crash": True,
            "reason": "native_timeout: verify_terminal exceeded %.1fs" % timeout,
        }
    exitcode = process.exitcode
    if exitcode is not None and exitcode < 0:
        name = _signal_name(-exitcode)
        return {
            "ok": False, "crash": True, "signal": -exitcode,
            "signal_name": name,
            "reason": "native_crash: verify_terminal child terminated by %s"
                      % name,
        }
    return {
        "ok": False, "crash": True,
        "reason": "native_crash: verify_terminal child exited with code %r "
                  "without a verdict" % (exitcode,),
    }


__all__ = [
    "AnswerShapeError", "SHAPE_EXCEPTIONS", "normalize_answer",
    "verify_isolated",
]