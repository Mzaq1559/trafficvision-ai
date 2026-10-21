# Audit and development notes

## Issues addressed in this branch
- Added the missing typed configuration module referenced by `app.config` and `app.pipeline`.
- Added a minimal Streamlit local-video entry point using the existing detector/pipeline/drawing modules.
- Replaced the placeholder README with architecture, verified capabilities, methodology, limitations, setup, evaluation guidance, and references.
- Added lightweight tests for configuration, trajectory, and speed utilities.

## Remaining validation
- The application and tests must be run in a clean checkout with dependencies installed.
- No labelled traffic dataset or benchmark results are included.
- The Streamlit workflow should be exercised with a real sample video; UI rendering and annotation compatibility depend on installed dependency versions.
- The current analytics semantics and global scale speed calibration remain research limitations, not fixed by documentation.
