"""Retain real control truth/CDC and zero-COMB checks, add native snapshot RAM."""
import json,sys
from pathlib import Path
from run_units import sha
from snapshot_ram_board import audit_delta
import audit_split_logic_outputs_map as checks
if __name__=='__main__':
 args=sys.argv[:]
 root=Path(args[args.index('--root')+1]).resolve()
 board=Path(args[args.index('--board')+1]).resolve()
 checks.audit_delta=audit_delta
 try:checks.main()
 finally:sys.argv=args
 from snapshot_ram_cdc import structure
 (root/'snapshot_ram_structure.json').write_text(json.dumps(structure(board),indent=2))
 (root/'ram_wrapper.json').write_text(json.dumps(dict(wrapper_sha256=sha(__file__),
  base_checker_sha256=sha(checks.__file__),scope='Existing control truth and finite probe CDC checks retained; new RAM structural checks added.'),indent=2))
