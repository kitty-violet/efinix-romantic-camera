"""Exercise release rejection and validate rebound warning evidence offline."""
import json
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import once

def main():
    source=HERE/'preflight_native_release.py';code=source.read_text()
    for before,after in (
        ('import finish_native_split_release as release','import finish_local_driver_release as release'),
        ('native-package-preflight1','local-driver-package-preflight1'),
        ('native_transform_sha256','local_driver_packager_sha256'),
        ('must-not-exist-native-preflight','must-not-exist-local-driver-preflight'),
        ('NATIVE_RELEASE_STATIC_AND_FAILED_STA_REJECTION_PASS','LOCAL_DRIVER_RELEASE_STATIC_AND_FAILED_STA_REJECTION_PASS')):
        code=once(code,before,after)
    scope=dict(__file__=__file__,__name__='local_driver_preflight')
    exec(compile(code,str(source)+':local-driver','exec'),scope);scope['main']()
    import finish_local_driver_release as release
    root=ROOT/'tmp/board-test/local-driver-package-preflight1'
    report=release.warning_audit(release.BOARD)
    for f,h in report['evidence'].items():assert sha(f)==h,f
    (root/'warning_audit.json').write_text(json.dumps(report,indent=2))
    r=json.loads((root/'result.json').read_text())
    r.update(warning_audit_sha256=sha(root/'warning_audit.json'),template_sha256=sha(source))
    (root/'result.json').write_text(json.dumps(r,indent=2))
    print('CURRENT_WARNING_LINEAGE_VALIDATED')

if __name__=='__main__':main()
