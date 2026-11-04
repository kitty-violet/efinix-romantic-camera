"""Existing bounded read-only snapshot worker, bound to the new run-page schema."""
import read_camera_checked_snapshot as base
import read_camera_checked_snapshot_v2 as runner
from read_cnn_run_progress import decode
base.decode=decode
original=runner.checked_manifest
def checked_manifest(path):
 m=original(path)
 assert m['cnn_run_contract']==dict(deadline_cycles=33554432,counter_bits=26,hardware_frame_limit=16,progress_schema='D-run-progress-111')
 return m
runner.checked_manifest=checked_manifest
runner.__file__=__file__
if __name__=='__main__':runner.main()
