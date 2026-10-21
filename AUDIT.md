# Audit and development notes

This file records the independent review findings and the verification status on
`docs/verify-and-license`. Test results below are the latest reported results
for the branch; the real YOLO-backed application was not exercised end-to-end.

## Fixed and covered by regression tests

- **Duplicate tracker updates on skipped frames.** The detector and ByteTrack
  update only on inference frames; skipped frames return the last-known state.
  See `app/pipeline.py` and `tests/test_pipeline_tracking.py`.
- **Per-frame results held live track references.** Processed frames now contain
  copies of track state, so reading earlier results after processing later frames
  does not mutate their trajectories or status.
- **Lost tracks rendered as active.** Lost tracks remain in the retained-track
  list for summary purposes but are excluded from drawing and active-speed
  statistics.
- **Edge-clipped boxes distorted speed estimates.** Samples are skipped when
  either of the two boxes used for the estimate touches a frame border; the last
  reliable speed is retained. Tests cover all four edges and fully visible
  motion. This is a mitigation, not perspective correction or a guarantee of
  accurate real-world speed.
- **`speed.enabled` was ignored.** The pipeline now honours it.
- **Missing or malformed explicit config paths silently fell back to defaults.**
  Explicit config errors are reported; absence of the default config may use
  built-in defaults with a warning. Unknown YAML keys produce warnings.
- **Empty detection class lists silently selected defaults.** Empty lists are
  rejected; `None` retains the documented default class behavior.
- **Streamlit results disappeared on rerun.** Results are stored in session
  state and invalidated when the upload changes or is removed. New processing
  replaces prior results.
- **Plain pytest depended on invocation context.** `pytest.ini` configures the
  repository path and test config fixtures resolve from the test file location.
  Both `pytest` and `python -m pytest` were reported passing from the repo root;
  collection was also checked from other working directories.
- **Partial URL downloads poisoned the cache.** Downloads use temporary files
  and only replace the cache after successful validation. Failure cleanup and
  preservation of a previous valid cache are tested.
- **Failed webcam opens leaked a capture handle.** The capture is released on
  failure and covered by a test.
- **Test suite did not cover synthetic end-to-end speed behavior.** Generated
  videos and an injected threshold detector now verify calibrated ground-truth
  speed, stable track identity, trajectory cadence, and duplicate-point
  prevention without YOLO weights.
- **Test dependency guards could silently skip core tests.** The test suite uses
  explicit lightweight dependencies in `requirements-test.txt`; the tests no
  longer hide missing core imports behind `importorskip`.
- **UI mislabeled tracker updates as frames seen.** The table reports the
  inference-update count as `frames_seen`; the UI metric separately reports
  the number of video frames processed.
- **Analytics and smoothing documentation were inaccurate.** The analytics
  docstring and effective minimum smoothing-window behavior are documented.
- **Ruff status was overstated.** `ruff.toml` defines the project policy
  explicitly: E/F/W checks, with E501 ignored. The README documents this exact
  policy instead of claiming the full default Ruff rules are clean.

## Remaining limitations and risks

- **Real YOLO path not verified.** PyTorch, Ultralytics weights, and a full
  real-video Streamlit session were not exercised during the lightweight test
  run. The synthetic detector is not a substitute for evaluating detector
  accuracy or tracking quality on real traffic footage.
- **Shared detector concurrency is unverified.** The model is cached across
  Streamlit reruns/sessions with `st.cache_resource`; concurrent inference
  safety has not been established.
- **Video source abstractions are not wired into the UI.** Webcam and URL
  source classes exist in `app/video/source.py`, but the current UI accepts a
  local upload only.
- **Track IDs are temporary and may be process-wide.** Do not interpret them as
  stable identities across separate videos or runs.
- **Lost tracks remain in memory for the duration of a run.** Long-run memory
  behavior has not been benchmarked.
- **ByteTrack frame rate is fixed by config.** It is not derived from input-video
  FPS; the lost-track buffer is measured in inference updates.
- **Speed calibration is a single global scale.** There is no perspective
  correction or road-plane homography. Border clipping is handled conservatively,
  but partially occluded objects and detector jitter can still affect estimates.
- **No line-crossing counts or ground-truth metrics are included.** Retained-track
  totals are not unique-vehicle ground truth or line-crossing counts. No labelled
  dataset, mAP, HOTA, IDF1, counting error, or speed-error benchmark is provided.
- **Reserved configuration options remain unused.** Processing-mode presets,
  `frame_skip`, selected UI refresh/table/theme settings, and the `demo` section
  do not change runtime behavior.
- **No dependency lockfile or platform matrix.** A dependency dry run is not the
  same as validating a full installation on all supported operating systems.
- **Commit-message typo.** An earlier commit message contains `feed:` where
  `fix:` was intended. Existing history was not rewritten.

## Validation reported for this branch

- `pytest -q`: 63 passed.
- `python -m pytest -q`: 63 passed.
- Test collection was checked from `/tmp` and `tests/`.
- `ruff check app streamlit_app.py tests`: clean under the configured E/F/W
  policy with E501 ignored.
- `mypy app --ignore-missing-imports`: clean.
- `python -m compileall`: clean.
- Synthetic-video regression tests match the expected 36.0 km/h for the tested
  calibrated motion and catch edge-clipping and duplicate-trajectory regressions.
- Interrupted-download tests confirm failed transfers leave no partial cache
  file and do not overwrite a previously valid cached video.
- Mutation checks were reported to catch the deliberately reintroduced edge
  check, snapshot, and duplicate-update regressions.

These results describe the checks actually reported for this branch. They do
not establish end-to-end accuracy, performance, or safety on real traffic video.
