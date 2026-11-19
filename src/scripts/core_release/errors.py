"""Typed errors shared by the release assembler and its validators."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Violation:
    """One machine-checkable contract failure.

    ``code`` is a stable token tests and operators key on; ``where`` is the
    pack- or release-relative location; ``message`` is human detail.
    """

    code: str
    where: str
    message: str

    def describe(self) -> str:
        return f"{self.code} [{self.where}] {self.message}"


class PackContractError(Exception):
    """Raised when one or more input world packs violate the pack contract."""

    def __init__(self, violations: list[Violation]) -> None:
        if not violations:
            raise ValueError("PackContractError requires at least one violation")
        self.violations = tuple(violations)
        detail = "\n".join(f"  {violation.describe()}" for violation in self.violations)
        super().__init__(f"{len(self.violations)} contract violation(s):\n{detail}")


class ReleaseError(Exception):
    """Raised when the release itself cannot be assembled or verified."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")