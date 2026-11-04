"""Freeze RAM observation candidate only after isolated and real bridge gates."""
import json,subprocess,sys
from run_units import HERE,ROOT
from snapshot_ram_board import BASE,BOARD,VAR,proofs,audit_delta,parent_audit
def main():
 parent_audit(BASE);proofs();assert not BOARD.exists()
 subprocess.run([sys.executable,'-B',str(HERE/'prepare_board_variant.py'),'--base',str(BASE),
  '--root',str(BOARD),'--override',str(VAR)],cwd=ROOT,check=True,timeout=60)
 result=audit_delta(BOARD)
 (BOARD/'snapshot_ram_binding.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
