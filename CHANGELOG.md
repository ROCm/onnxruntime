# Change Log for ROCm onnxruntime

This ROCm fork of onnxruntime is only used for staging pull request branches and maintaining rocmX.Y_internal_testing branches. New ROCm features are staged into the testing branches for internal QA teams to validate against the respective ROCm release before they are upstreamed. Testing branches are created from a known good tip-of-tree upstream main commit near the point in time a ROCm release is branched. Keeping the testing branches mostly unchanging aids the QA team in regression testing.

This CHANGELOG will only indicate features that were staged to the testing branch during the corresponding ROCm release process.

## onnxruntime for ROCm 5.5

### Added
- Modify MIGraphX EP for Accuracy tests
- added new miopenGetConvolutionSpatialDim api and unit test passed

### Fixed
- work around -Werror and -Wdeprecated-builtins in abseil-cpp
- Patch eval_squad.py script for Python < 3.8 and multiple Execution Providers
- ROCm header path warning fixes
