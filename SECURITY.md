# Security policy

## What is a security issue here

* A way to **recover the payload or the secret table without the keyfile**
  (from weights, from black-box queries, or from published evaluation reports).
* A way to **forge** a `WATERMARK_PRESENT` verdict for a model that does not
  carry the watermark (false-positive attack on the verifier / statistics).
* A way to **remove** the watermark that is not already listed as an attack in
  `docs/EVALUATION.md` / `docs/THREAT_MODEL.md` while keeping the model useful —
  we track these as research findings; please still report them privately first
  if they affect deployed models.
* Leakage of key material through the tooling (logs, reports, sidecar files).

Not security issues: model quality regressions, attacks that are explicitly
listed as out of scope (distillation, collusion averaging, targeted unlearning
with white-box access), or the inherent fact that the model asserts the
fictitious facts are real (documented limitation).

## Reporting

Please **do not** open a public issue for the above. Use GitHub's private
vulnerability reporting ("Report a vulnerability" under the Security tab of
https://github.com/rickgwei/llm-distill-watermark) or contact the maintainer
through the profile linked there. Include a minimal reproduction; if it needs a
keyfile, generate a throw-away one with `ldw keygen --no-registry` — never send
a production key.

You will get an acknowledgement within a week. Fixes and advisories are
published in `CHANGELOG.md` and as GitHub security advisories.

## Handling keys

* Keyfiles are secrets. Store them encrypted (`ldw keygen --passphrase …`) and
  outside the repository; `.gitignore` excludes `*.keyfile.json` and the
  pre-commit hook refuses files containing `master_key_hex`.
* Publish only redacted evaluation reports (`ldw eval export`), never the
  `items/` directory or raw `verify` reports of a production key — per-probe
  answers reveal entity names and codebook values.
* The commitment (`sha256` of the secret table) can be published or timestamped
  at keygen time to prove precedence later without revealing the key.

## Supported versions

The latest minor release receives fixes; the project is pre-1.0 (`0.x`) and
minor versions may change on-disk formats (documented in the changelog).
