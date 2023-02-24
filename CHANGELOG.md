# Change Log for ROCm onnxruntime

This ROCm fork of onnxruntime is only used for staging pull request branches and maintaining rocmX.Y_internal_testing branches. New ROCm features are staged into the testing branches for internal QA teams to validate against the respective ROCm release before they are upstreamed. Testing branches are created from a known good tip-of-tree upstream main commit near the point in time a ROCm release is branched. Keeping the testing branches mostly unchanging aids the QA team in regression testing.

This CHANGELOG will only indicate features that were staged to the testing branch during the corresponding ROCm release process.

## onnxruntime for ROCm 5.6

### Added
- Added new miopenGetConvolutionDescriptorSize api and unit test passed.
  This was previously part of ROCm 5.5, but upstreaming was not complete by the time ROCm 5.6 was branched.
