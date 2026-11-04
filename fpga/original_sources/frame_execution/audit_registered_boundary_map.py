"""Use the unchanged actual-map truth and CDC checks on this exact delta."""
import json,sys
from pathlib import Path
from run_units import sha
from registered_boundary_board import audit_delta
import audit_split_logic_outputs_map as checks

if __name__=='__main__':
    checks.audit_delta=audit_delta
    root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
    checks.main()
    (root/'registered_wrapper.json').write_text(json.dumps(dict(
        wrapper_sha256=sha(__file__),base_checker_sha256=sha(checks.__file__),
        scope='Only candidate delta audit changed; all mapped truth, FF and CDC checks retained.',hardware_access=False),indent=2))
