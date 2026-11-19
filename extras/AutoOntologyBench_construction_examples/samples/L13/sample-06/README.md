# L13 construction example: L13-construction-01

- task_id: `l13-in-03cc66537ba22b17d661`
- family_id: `l13-x07-union-response-construction`
- regime: `construction`
- difficulty: `structural-high`
- oracle_handle: `l13x-oracle-4edd89b5d4681c0a93c9`
- source snapshots: `apis-guru@commit-f04b8d0bcd39`

## Public goal

Construct a JSON instance that validates against the 200 response schema of GET /api/auths/{id} in the pinned OpenAPI spec 'Otoroshi Admin API' (maif.local/otoroshi 1.5.0-dev). The schema is a union (oneOf/anyOf); the instance must validate against exactly one branch.

## Budget / tools

- allowed_tools: `openapi_spec_inspect, contract_validate`
- budget: `{"actions": 14, "probes": 8}`

## Public observations

- `external_knowledge/pinned/apis-guru/commit-f04b8d0bcd39/APIs/maif.local/otoroshi/1.5.0-dev/openapi.yaml`

## Answer expectation

Return a JSON instance that validates against the `200` response schema of `GET /api/auths/{id}` and matches exactly one `oneOf` branch.
