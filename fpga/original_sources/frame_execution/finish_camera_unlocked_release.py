"""Original full hardware release plus corrected camera host deadline evidence."""
from run_units import HERE
from control_command_repair import once
body=(HERE/'finish_camera_checked_release_v3.py').read_text()
body=body.replace('finish_camera_checked_release_v3','finish_camera_unlocked_release')
prefix,guard,tail=body.rpartition("if __name__=='__main__':main()")
assert guard and not tail.strip()
exec(compile(prefix,__file__+':original-strict-hardware-gates','exec'),globals())
unlocked_camera_evidence=camera_evidence
def camera_evidence(bind,verify):
    r=unlocked_camera_evidence(bind,verify)
    b=bind(ROOT/'tmp/board-test/camera-reader-budget1/result.json')
    assert b['status']=='CAMERA_READER_SEPARATE_BUDGET_TESTS_PASS' and b['tests']==9
    verify(b['evidence']);r['separate_budget_checks']=b['tests'];return r
unlocked_package=package_code
def package_code():
    source,code=unlocked_package()
    code=once(code,'from read_camera_checked_snapshot import main','from read_camera_checked_snapshot_v2 import main')
    code=code.replace("HERE/'CAMERA_CHECKED_RELEASE.md',out/'README.md'","HERE/'CAMERA_CHECKED_RELEASE_C3.md',out/'README.md'")
    code=once(code,"    for f in out.glob('*.py'):compile", "    verify({str(HERE/'CAMERA_CHECKED_RELEASE_C3.md'):sha(HERE/'CAMERA_CHECKED_RELEASE_C3.md')})\n    for f in out.glob('*.py'):compile")
    compile(code,'unlocked-camera-package-preflight','exec');return source,code
if __name__=='__main__':main()
