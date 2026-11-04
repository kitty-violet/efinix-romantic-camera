"""Same hardware release, with validated separate host verification/IO budgets."""
from run_units import HERE
from control_command_repair import once
body=(HERE/'finish_camera_checked_release_v7.py').read_text()
body=once(body,"'finish_camera_checked_release_v7'","'finish_camera_checked_release_v8'")
prefix,guard,tail=body.rpartition("if __name__=='__main__':main()")
assert guard and not tail.strip()
exec(compile(prefix,__file__+':v7-hardware-gates','exec'),globals())
previous_camera_evidence=camera_evidence
def camera_evidence(bind,verify):
    r=previous_camera_evidence(bind,verify)
    budget=bind(ROOT/'tmp/board-test/camera-reader-budget1/result.json')
    assert budget['status']=='CAMERA_READER_SEPARATE_BUDGET_TESTS_PASS' and budget['tests']==9
    verify(budget['evidence']);r['separate_budget_checks']=budget['tests'];return r
v8_original_package=package_code
def package_code():
    source,code=v8_original_package()
    code=once(code,'from read_camera_checked_snapshot import main','from read_camera_checked_snapshot_v2 import main')
    compile(code,'camera-v8-package-preflight','exec');return source,code
if __name__=='__main__':main()
