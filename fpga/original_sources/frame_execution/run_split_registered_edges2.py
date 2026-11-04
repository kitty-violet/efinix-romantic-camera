"""Use this installation's existing SALT license setting for boundary tests."""
import os
import validate_split_registered_edges as gate
os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
gate.RUN=gate.ROOT/'tmp/frame-execution/split-registered-edges-shorts2'
if __name__=='__main__':
 try:gate.main()
 except Exception as exc:
  if gate.RUN.exists():(gate.RUN/'failure.json').write_text(gate.json.dumps(dict(error=repr(exc)),indent=2))
  raise
