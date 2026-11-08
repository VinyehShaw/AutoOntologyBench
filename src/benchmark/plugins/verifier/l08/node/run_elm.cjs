#!/usr/bin/env node
/* L08 ELM batch runner: executes ELM libraries over FHIR R4 patient bundles.
 *
 * stdin/arg: job file JSON:
 *   {"jobs": [{"job_id": str, "library": <ELM JSON>, "patients": [<FHIR Bundle>...]}]}
 * stdout: {"engine": {...}, "results": [{"job_id": str, "ok": bool,
 *           "patient_results": {pid: {expr: value}}, "error": str?}]}
 *
 * Deterministic: no wall-clock expressions are used by lane ELM (Today/Now are
 * never emitted); MeasurementPeriod is an ELM parameter with a fixed default.
 */
'use strict';

const fs = require('fs');
const cql = require('cql-execution');
const cqlfhir = require('cql-exec-fhir');

function norm(v, depth) {
  if (v === null || v === undefined) return null;
  const d = depth || 0;
  if (d > 24) return String(v);
  if (typeof v === 'boolean' || typeof v === 'number' || typeof v === 'string') return v;
  if (Array.isArray(v)) return v.map(x => norm(x, d + 1));
  if (v.isDateTime) return {__type: 'DateTime', iso: v.toString()};
  if (v.isDate) return {__type: 'Date', iso: v.toString()};
  if (v.isQuantity) return {__type: 'Quantity', value: v.value, unit: v.unit};
  if (v.isInterval) return {__type: 'Interval', low: norm(v.low, d + 1), high: norm(v.high, d + 1)};
  if (v.isConcept) return {__type: 'Concept', codes: (v.codes || []).map(c => norm(c, d + 1))};
  if (v.isCode) return {__type: 'Code', system: v.system, code: v.code};
  if (typeof v === 'object' && v.constructor && v.constructor.name === 'FHIRObject') {
    return {__type: 'FHIRObject', resourceType: v.getType && v.getType()};
  }
  if (typeof v === 'object') {
    const out = {};
    for (const k of Object.keys(v)) out[k] = norm(v[k], d + 1);
    return out;
  }
  return String(v);
}

async function runJob(job) {
  const lib = new cql.Library(job.library);
  const executor = new cql.Executor(lib);
  const psource = new cqlfhir.PatientSource.FHIRv401();
  psource.loadBundles(job.patients);
  const result = await executor.exec(psource);
  const patientResults = {};
  for (const [pid, exprs] of Object.entries(result.patientResults || {})) {
    const clean = {};
    for (const [name, val] of Object.entries(exprs)) {
      if (name === 'Patient') continue;
      clean[name] = norm(val, 0);
    }
    patientResults[pid] = clean;
  }
  return {job_id: job.job_id, ok: true, patient_results: patientResults};
}

async function main() {
  const jobPath = process.argv[2];
  const spec = JSON.parse(fs.readFileSync(jobPath, 'utf8'));
  const out = {
    engine: {
      node: process.version,
      'cql-execution': require('cql-execution/package.json').version,
      'cql-exec-fhir': require('cql-exec-fhir/package.json').version,
    },
    results: [],
  };
  for (const job of spec.jobs) {
    try {
      out.results.push(await runJob(job));
    } catch (err) {
      out.results.push({job_id: job.job_id, ok: false, error: String((err && err.stack) || err)});
    }
  }
  process.stdout.write(JSON.stringify(out));
}

main().catch(err => {
  process.stderr.write(String((err && err.stack) || err));
  process.exit(2);
});
