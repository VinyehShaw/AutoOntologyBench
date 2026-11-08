# Public eval tests

The private oracle fixtures and the two tests that require them are not shipped
in the public code package:

- `fixtures/oracle/`
- `test_oracle.py`
- `test_smoke_l05.py`

The remaining tests cover task loading, adapters, providers, scoring, metrics,
and the offline reporter. To run oracle-backed smoke tests, mount the private
oracle separately and use the internal test fixtures.