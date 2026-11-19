# L04 construction example: L04-construction-01

- task_id: `l04-01c36bab72a0-232f2344`
- family_id: `l04.con_ancestor_closure_via_probes`
- regime: `construction`
- difficulty: `structural-high`
- oracle_handle: `l04-oracle-b78443f7c1e8a341`
- source snapshots: `rhea-chebi/rhea-142+chebi-256`

## Public goal

Only the label of ChEBI entity CHEBI:131450 (2,6-dihydroxybenzoate) is public, plus a budgeted probe tool that reveals the direct rdfs:subClassOf parents of one ChEBI class per probe within the stated budget. Construct the transitive ancestor closure yourself and report the exact, complete, sorted set of ancestor ChEBI class identifiers of CHEBI:131450 in the pinned ChEBI ontology (excluding CHEBI:131450 itself).

## Budget / tools

- allowed_tools: `inspect_schema, inspect_state, read_public_doc, probe_native, edit_artifact, validate_artifact, check_action_with_ontology, submit_final`
- budget: `{"actions": 4, "probes": 26}`

## Public observations

- `public/l04-01c36bab72a0-232f2344/chebi_entity.json`
- `public/l04-01c36bab72a0-232f2344/initial_state.json`
- `public/l04-01c36bab72a0-232f2344/probe_spec.json`

## Answer expectation

Return the exact, complete, sorted set of ancestor ChEBI class identifiers of `CHEBI:131450` (excluding `CHEBI:131450` itself).
