"""Answer-level evaluator for the AutoOntologyBench Core-track release.

The Core-track release is a benchmark packaging of 16 world packs. This
package provides a deliberately small, auditable evaluation path:

* load the public Core task rows;
* optionally ask an OpenAI-compatible/vLLM/Anthropic model for one answer per
  task (``model_runner``);
* score those answers against a separately mounted private Core oracle
  (``run``).

It does **not** claim to be the full ontology-repair harness. The Core release
does not ship the private gold bundles, the frozen executor, or the source-pack
toolchain. A consumer that needs the full agentic loop should supply its own
harness and emit the same submission schema.
"""

from __future__ import annotations

__all__ = ["__version__", "REPORT_SCHEMA", "SUBMISSION_SCHEMA"]

__version__ = "0.1.0"

#: Schema tag of the evaluation report written by ``benchmark.eval_core.run``.
REPORT_SCHEMA = "aob-eval-core/1"

#: Schema tag accepted for Core submission files.
SUBMISSION_SCHEMA = "aob-submission-core/1"