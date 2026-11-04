"""Re-audit completed immutable STA reports after fixing one CDC classifier.

Retain original failed receipts. No Efinity run, route or constraints change.
"""
import json,re,subprocess,sys,time
from pathlib import Path
from run_units import ROOT,HERE,sha
from camera_checked_cdc_v3 import inventory
from audit_snapshot_jtag_paths import audit as audit_jtag

def main():
 batch=ROOT/'tmp/frame-execution/camera-ready-route1';r=batch/'gray-seed3'
 assert not (r/'result.json').exists() and not (batch/'result.json').exists()
 proof=ROOT/'tmp/board-test/camera-retained-done-v3-proof1.json';p=json.loads(proof.read_text())
 assert p['status']=='EXACT_RETAINED_DONE_CONE_AND_NEGATIVE_CHECKS_PASS' and len(p['negative_checks'])==3
 for f,h in p['evidence'].items():assert sha(f)==h,f
 mi=json.loads((r/'inputs.json').read_text());original=dict(mi['original']);original.update(mi['scripts'])
 for f,h in original.items():assert sha(f)==h,f
 physical=Path(mi['base']);assert mi['canonical_acceptance_unchanged']
 assert sha(r/'ti60f225_oob.sdc')==mi['diagnostic_sdc_sha256'] and sha(r/'gray.tcl')==mi['tcl_sha256']
 for suffix in ('.troutingtraces','.place','.vdb','.map.v'):
  f=physical/'outflow'/('ti60f225_oob'+suffix);assert sha(r/'outflow'/f.name)==original[str(f)]
 failure=json.loads((r/'failure.json').read_text());assert 'inventory' in failure['traceback'] and failure['error']=='AssertionError()'
 stages=json.loads((batch/'stages.json').read_text());assert [(s['name'],s['exit_code']) for s in stages]==[('route',0),('canonical',0),('gray',1)]
 selected=json.loads((batch/'selected.json').read_text());assert selected['current_timing_pass'] and selected['wns_ns']>=0 and selected['whs_ns']>=0
 logtext=(r/'sta.log').read_text(errors='replace')+'\n'+'\n'.join(f.read_text(errors='replace') for f in (r/'outflow').glob('*.sta.out'))
 assert not re.search(r'Unable to run|No (?:ports|pins|clocks) matched|Timer cuts a combinational loop|^\s*(?:ERROR|FATAL)\s*:',logtext,re.M)
 reports=list((r/'outflow').glob('*.rpt'))+list((r/'outflow').glob('*.sta.out'))
 evidence={str(f):sha(f) for f in reports+[r/'inputs.json',r/'sta.log',r/'failure.json',batch/'failure.json',batch/'stages.json',batch/'selected.json',proof,HERE/'camera_checked_cdc_v3.py',Path(__file__)]};evidence.update(original);evidence.update(p['evidence'])
 receipt=r/'reaudit_inputs.json';assert not receipt.exists()
 receipt.write_text(json.dumps(dict(status='RETAINED_DONE_CLASSIFIER_REAUDIT_INPUTS',evidence=evidence,hardware_or_constraint_change=False,sta_rerun=False,hardware_access=False),indent=2))
 start=time.monotonic();inv=inventory(r);jtag=audit_jtag(r)
 (r/'clock_crossing_inventory.json').write_text(json.dumps(inv,indent=2)+'\n');(r/'jtag_cdc_audit.json').write_text(json.dumps(jtag,indent=2))
 check=subprocess.run([sys.executable,'-B',str(HERE/'audit_gray_skew_result.py'),'--board',str(r)],capture_output=True,text=True,timeout=30)
 assert check.returncode==0,check.stdout+check.stderr
 skew=json.loads((r/'gray_skew_audit.json').read_text())
 for f,h in evidence.items():assert sha(f)==h,f
 rr=dict(status='COMPLETED_REPORTS_CDC_REAUDIT_PASS',inputs_sha256=sha(receipt),seconds=time.monotonic()-start,inventory_sha256=sha(r/'clock_crossing_inventory.json'),evidence_unchanged=True,old_failure_retained=True,route_rerun=False,hardware_access=False)
 (r/'reaudit_result.json').write_text(json.dumps(rr,indent=2))
 result=dict(status='NEW_ROUTE_GRAY_SKEW_PASS_NOT_FULL_CDC_RELEASE',buses=skew['buses'],endpoints=skew['endpoints'],max_skew_ns=skew['max_skew_ns'],limit_ns=skew['limit_ns'],inputs_sha256=sha(r/'inputs.json'),inventory_sha256=sha(r/'clock_crossing_inventory.json'),jtag_audit_sha256=sha(r/'jtag_cdc_audit.json'),skew_audit_sha256=sha(r/'gray_skew_audit.json'),reaudit_sha256=sha(r/'reaudit_result.json'),original_and_route_unchanged=True,diagnostic_interpll_setup_not_acceptance=True,cdc_release=False,board_release=False)
 (r/'result.json').write_text(json.dumps(result,indent=2))
 (batch/'result.json').write_text(json.dumps(dict(status='LOCAL_DECODE_ROUTED_ORIGINAL_STA_GRAY_PASS_PENDING_RELEASE_AUDITS',selected=selected,iterations=49,gray_reaudit=str(r/'reaudit_result.json'),gray_reaudit_sha256=sha(r/'reaudit_result.json'),original_failure_retained=True,hardware_access=False,board_release=False),indent=2))
 print(json.dumps(result))
if __name__=='__main__':main()
