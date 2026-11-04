"""Existing bounded reader, gated by the exact reduced-observer manifest."""
import read_camera_checked_snapshot as base
import read_cnn_run_snapshot as original
import read_camera_checked_snapshot_v2 as runner
from read_cnn_observer_trim import decode,PROFILE
base.decode=decode
previous=runner.checked_manifest

def checked_manifest(path):
    result=previous(path)
    assert result['observer_diagnostic_profile']==PROFILE
    return result

runner.checked_manifest=checked_manifest
runner.__file__=__file__
if __name__=='__main__':runner.main()
