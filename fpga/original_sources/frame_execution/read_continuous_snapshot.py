"""Existing bounded read-only worker with the current continuous manifest."""
import read_camera_checked_snapshot as base
import read_camera_checked_snapshot_v2 as runner
from read_continuous_progress import decode,PROFILE
base.decode=decode
previous=runner.checked_manifest

def checked_manifest(path):
 m=previous(path)
 assert m['observer_diagnostic_profile']==PROFILE
 c=m['cnn_run_contract']
 assert c['hardware_frame_limit']==512 and c['frame_budget_bits']==10
 assert c['boot_run_watchdog_cycles']==9000000000
 assert c['explicit_success_rearm'] and c['full_reload_on_rearm']
 assert c['progress_schema']=='D-continuous-progress-v1'
 return m
runner.checked_manifest=checked_manifest
runner.__file__=__file__
if __name__=='__main__':runner.main()
