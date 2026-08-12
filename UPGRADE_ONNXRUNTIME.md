# Supporting a New ONNX Runtime Version

This is the end-to-end runbook for advancing the repository to a new pinned
ONNX Runtime (ORT) release, from the version bump to the published
`ort-v<version>` tag.

The dataset under `artifacts/` is derived from ORT's own test suite at one
pinned upstream release. "Supporting a new ORT version" therefore means:
move the pin, let CI re-extract the dataset, repair whatever the new upstream
release broke in the extractor, and tag the result.

## TL;DR

1. Edit three files by hand: `requirements.txt`, `artifacts/MANIFEST.json`,
   `tests/test_onnxruntime_source.py`.
2. Push to a branch and open a pull request.
3. `Refresh Artifact Dataset` CI re-extracts the dataset and pushes a
   `chore: refresh ORT artifact dataset` commit onto the pull request branch.
4. Fix any extractor breakage the new ORT release introduced; repeat until CI
   is green.
5. Squash-merge. CI then publishes the `ort-v<version>` tag and release
   automatically.

## Background: What Drives the Version

Everything derives from the single exact pin in `requirements.txt`:

```text
onnxruntime==<major>.<minor>.<patch>
```

`tools/python/emx_ort_test_materializer/onnxruntime_source.py` reads that pin
(`read_pinned_onnxruntime_version`), maps it to the upstream Git tag
`v<major>.<minor>.<patch>` (`onnxruntime_version_tag`), and clones or updates
an ephemeral checkout under `build/onnxruntime-org` at that tag. Never
hard-code a commit, branch, or tag anywhere else.

## Step 1: Pick the Target Version

Choose an exact `onnxruntime` release published on PyPI that also has a
matching `v<version>` release tag in
[microsoft/onnxruntime](https://github.com/microsoft/onnxruntime/tags).
Both must exist — the PyPI package is installed for validation, the Git tag is
checked out for extraction. Pre-releases and release candidates are not
supported by the pin pattern.

## Step 2: Move the Pin (Hand-Edited Files)

Exactly three files are edited by hand for a plain version bump. Replace the
old version everywhere it appears:

| File | Change |
| --- | --- |
| `requirements.txt` | `onnxruntime==<old>` → `onnxruntime==<new>` |
| `artifacts/MANIFEST.json` | `"pinned_onnxruntime_version": "<new>"` |
| `tests/test_onnxruntime_source.py` | both version assertions: `read_pinned_onnxruntime_version(...) == "<new>"` and `onnxruntime_version_tag("<new>") == "v<new>"` |

A quick sanity check that nothing was missed:

```bash
grep -rn "<old-version>" --exclude-dir=artifacts --exclude-dir=.git .
```

`artifacts/MANIFEST.json` is the one file under `artifacts/` that is
maintained by hand; the rest of that tree is generated.

## Step 3: Push and Let CI Refresh the Dataset

Commit the pin change (for example `chore: bump ONNX Runtime pin to v<new>`),
push the branch, and open a pull request.

`.github/workflows/refresh-artifact-dataset.yml` runs on every pull request
push and does the expensive work:

1. Deletes `artifacts/onnxruntime` and `artifacts/onnxruntime-negative`.
2. Runs `python tools/scripts/extract_test_artifacts.py --artifacts-output artifacts --rebuild`,
   which clones ORT at the pinned tag, builds the runtime extractor against
   ORT's test sources, and replays the tests to capture artifacts.
3. Runs `python tools/scripts/update_artifact_validation_expectations.py`.
4. Regenerates `artifacts/VALIDATION_ERRORS.md` and `artifacts/OPERATORS.md`.
5. Runs `pytest -q`.
6. Commits the regenerated dataset as `chore: refresh ORT artifact dataset`
   and pushes it back onto the pull request branch.

**Do not run the full extraction locally to produce the committed dataset.**
It builds a large part of ORT's C++ test suite and takes hours; CI has the
ccache warmed from `main`. Local runs (see [`DEVELOPMENT.md`](DEVELOPMENT.md))
are for debugging a specific failure, ideally narrowed with a gtest filter.

Because CI pushes to the branch, always `git pull --rebase origin <branch>`
before making further edits, or the next push is rejected.

Two limitations worth knowing:

- Pull requests from forks do not get the auto-push; the branch must live in
  this repository.
- On pushes to `main` the workflow runs in `WARM_CCACHE_ONLY` mode: it only
  warms the compiler cache and commits nothing.

## Step 4: Repair What the New ORT Release Broke

This is the part that actually takes judgement. A new upstream release
regularly changes test sources, helper headers, or build requirements in ways
the extractor has to follow. Read the CI log, fix the cause, add a regression
test, push, and let CI re-run.

Real examples from previous upgrades:

- **New helper include the CI build cannot satisfy.** For 1.25.1, contrib test
  sources started including `contrib_ops/webgpu/...` headers that need a
  webgpu-enabled build. Fix: skip those helper sources in
  `helper_source_files()` in `tools/scripts/extract_test_artifacts.py`, plus a
  unit test in `tests/test_extract_test_artifacts_script.py`.
- **Test cases hidden behind gtest's `DISABLED_` prefix.** Also 1.25.1:
  extraction now passes `--gtest_also_run_disabled_tests`, and
  `StripDisabledGtestPrefix()` in
  `tools/cpp/runtime_extractor/ort_runtime_capture.cpp` keeps artifact paths
  free of the prefix, covered by `tests/test_runtime_extractor_source.py`.
- **Changed build requirements.** For 1.26.0, ORT's CMake needed the C
  language enabled, so
  `tools/cpp/runtime_extractor/CMakeLists.txt` became
  `project(emx_ort_runtime_extractor C CXX)`.
- **Transient dependency downloads.** ORT resolves its dependency archives
  (abseil, re2, googletest, ...) over the network during the CMake configure
  step, so an interrupted or intercepted download fails the whole extraction
  with a `FetchContent` error. This is infrastructure noise, not an upstream
  change: `configure_runtime_extractor()` retries the configure step
  (`CONFIGURE_ATTEMPTS`), and CMake resumes the missing downloads because the
  download stamps are only written on success. If every attempt fails with the
  same download error, the runner's network egress is the problem — never
  work around it by disabling TLS verification.
- **Newly failing validation cases.** Prefer fixing the pipeline. If a case
  cannot be replayed faithfully, record it in
  `artifact_generation_ignored_cases.json` with a concrete `reason`. Expected
  validation *outcomes*, by contrast, live in
  `tests/artifact_validation_expected.json` and are regenerated by CI — never
  hand-edit that file to silence a failure.

When touching extractor or tooling code, keep the layering rules from
[`AGENTS.md`](AGENTS.md) intact: generation policy stays in
generation/extraction code, validation only validates what exists, and
reporting only presents configured policy.

## Step 5: Review the Dataset Delta

Before merging, sanity-check the regenerated dataset rather than only the
green checkmark:

- `artifacts/MANIFEST.json` shows the new `pinned_onnxruntime_version`.
- `artifacts/OPERATORS.md` and `artifacts/VALIDATION_ERRORS.md` moved in a
  plausible direction — new operators and cases appearing is normal; a large
  unexplained drop in case count means extraction silently lost coverage.
- The non-OK case count in `artifacts/VALIDATION_ERRORS.md` did not grow
  without an explanation.

## Step 6: Merge — Tag and Release Are Automated

Squash-merge the pull request into `main`. That is the last manual step.

`.github/workflows/publish-ort-release.yml` runs on every push to `main`,
reads the pinned version from `requirements.txt`, and — if nothing has been
published for it yet — creates the tag `ort-v<version>` together with a GitHub
release named `ort-v<version>` and the body `Support ONNX Runtime v<version>`.
The `ort-` prefix names the supported ORT version, not a version of this
repository. Existing examples: `ort-v1.26.0`, `ort-v1.27.0`.

Two properties of the workflow matter:

- It listens only on pushes to `main`, so nothing can be published from a pull
  request branch. The tag appears exactly when the bump is squash-merged, and
  points at that merge commit.
- It is idempotent. Pushes that do not advance the pin find the release
  already present and exit. Tag and release are therefore created exactly once
  per ORT version, and an existing tag is never moved.

Consequence of the never-moved tag: later dataset refreshes for the same ORT
version land on `main` after the tag and are not contained in it. If such a
refresh has to be marked, publish it by hand.

Manual fallback, should the workflow fail or need re-running: trigger
`Publish ORT Release` via `workflow_dispatch`, or publish by hand on `main`
with
`gh release create ort-v<version> --title ort-v<version> --notes "Support ONNX Runtime v<version>"`.

## Never Hand-Edit

These are produced by the pipeline and committed by CI:

- everything under `artifacts/onnxruntime/` and
  `artifacts/onnxruntime-negative/` (`model.onnx`, `*.pb`, `validation.json`)
- `artifacts/OPERATORS.md`
- `artifacts/VALIDATION_ERRORS.md`
- `tests/artifact_validation_expected.json`

## Checklist

- [ ] `requirements.txt` pin updated
- [ ] `artifacts/MANIFEST.json` `pinned_onnxruntime_version` updated
- [ ] `tests/test_onnxruntime_source.py` assertions updated
- [ ] No stale occurrences of the old version outside `artifacts/`
- [ ] Branch pushed, pull request opened, `Refresh Artifact Dataset` green
- [ ] Extractor breakage fixed with regression tests, not by relaxing
      expectations
- [ ] Dataset delta reviewed (operator and validation reports)
- [ ] Squash-merged into `main`
- [ ] `Publish ORT Release` published the `ort-v<version>` tag and release on
      the merge commit
