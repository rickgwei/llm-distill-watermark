## Summary

<!-- What does this change and why? Link the issue if there is one. -->

## Checklist

- [ ] `make lint` and `make test` pass locally
- [ ] New/changed behaviour is covered by tests (offline tests use `FakeTarget`; model tests are marked `slow`)
- [ ] `tests/test_golden.py` still passes — or the change to the watermark algorithm / training data is intentional and documented in `CHANGELOG.md`
- [ ] Docs updated (`README.md`, `docs/`, `docs/CLI.md` for new commands, `docs/EVALUATION.md` for new metrics/datasets)
- [ ] `CHANGELOG.md` has an entry under *Unreleased*
- [ ] No keyfiles, model weights or per-item evaluation dumps are committed
