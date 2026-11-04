"""Reuse bounded full-board gates for the explicitly audited OFP5-only delta."""
import json,re,sys
from pathlib import Path
from run_units import HERE,sha
from startup_cause_board import BASE,audit_delta
def proof(board,folder):
 audit_delta(board);r=json.loads((folder/'result.json').read_text());i=json.loads((folder/'inputs.json').read_text())
 assert r['status']=='STARTUP_CAUSE_TRACE_CELLS_SERIAL_ADAPTER_PASS' and len(r['results'])==15
 assert r['inputs_sha256']==sha(folder/'inputs.json') and i['board_inputs_sha256']==sha(board/'inputs.json')
 assert i['runner_sha256']==sha(HERE/'validate_startup_cause.py') and i['decoder_sha256']==sha(HERE/'read_startup_cause.py')
 for n,h in i['files'].items():assert sha(folder/n)==h
 for x in r['results']:assert sha(folder/x['log'])==x['sha256'] and x['passes']
 return dict(path=str(folder),result_sha256=sha(folder/'result.json'))
def main():
 import fault_probe_flow as parent
 adjust_old=parent.adjust
 def adjust(code):
  code=adjust_old(code)
  code=code.replace('from read_fault_probe import decode','from read_startup_cause import decode')
  code=code.replace('[769:0]','[1025:0]').replace('[769:66]','[1025:66]')
  code=re.sub(r'(?<!\d)770(?!\d)','1026',code).replace("32'h4f465034","32'h4f465035").replace('raw[49:42]!==4','raw[49:42]!==5')
  code=code.replace("wire [703:0] payload={320'h123456789abcdef,ddr_lock_drops","wire [959:0] payload={576'h123456789abcdef,ddr_lock_drops")
  code=code.replace('len(selected)<7500','len(selected)<10000')
  return code
 parent.audit_delta=audit_delta;parent.BASE=BASE;parent.proof=proof;parent.adjust=adjust
 parent.main()
if __name__=='__main__':main()
