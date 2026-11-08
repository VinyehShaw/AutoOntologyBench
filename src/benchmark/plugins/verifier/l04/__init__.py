"""L04 verifier plugin: native rdflib SPARQL verification over the pinned graphs."""
from benchmark.plugins.verifier.l04.plugin import L04NativeVerifier

PLUGIN = L04NativeVerifier()

__all__ = ["L04NativeVerifier", "PLUGIN"]
