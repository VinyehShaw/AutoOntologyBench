"""Load-and-evaluate harness for the AutoOntologyBench 20K test release.

This package is the *evaluator side* of the 20K benchmark:

- :mod:`benchmark.eval_20k.tasks` loads and validates the public task file
  (``release_20k/tasks/qualified_tasks_test.jsonl``), or derives it from a
  private build tree with the same projection the release packager uses.
- :mod:`benchmark.eval_20k.submission` defines and validates the submission
  format (JSONL, one row per ``task_id``).
- :mod:`benchmark.eval_20k.oracle` resolves private oracle payloads from a
  mounted oracle root (never shipped with public tasks).
- :mod:`benchmark.eval_20k.lanes` dispatches per-task scoring to the existing
  per-lane ``NativeVerifierPlugin.verify_terminal`` implementations under
  ``benchmark.plugins.verifier``. Lanes that cannot be evaluated in the
  current environment are reported as ``NOT_SUPPORTED`` with an explicit
  reason; scores are never fabricated.
- :mod:`benchmark.eval_20k.scoring` aggregates per-task verdicts into TSR,
  V-TSR, unsafe counts/rates, cost/token totals and per-lane / per-family
  breakdowns.
- :mod:`benchmark.eval_20k.run` is the CLI entry point
  (``python -m benchmark.eval_20k.run``).
- :mod:`benchmark.eval_20k.model_runner` is an optional front-end that
  *generates* a submission JSONL from an OpenAI-compatible model endpoint;
  it never scores — scoring stays in :mod:`benchmark.eval_20k.run`.

No model answer is ever invented here: the runner records model output (or
an explicit error marker) verbatim, and the harness scores *given*
predictions against the mounted oracle.
"""
from __future__ import annotations

__all__ = ["__version__", "REPORT_SCHEMA", "SUBMISSION_SCHEMA"]

__version__ = "0.1.0"

#: Schema tag of the evaluation report written by the CLI.
REPORT_SCHEMA = "aob-eval-20k/1"

#: Schema tag accepted for submission files (first record may declare it).
SUBMISSION_SCHEMA = "aob-submission-20k/1"
