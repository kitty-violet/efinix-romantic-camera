"""Full original release gates plus current fixed-A arithmetic evidence. No I/O."""
import json
from pathlib import Path
from run_units import ROOT,HERE,sha
from accepted_tail_board import audit_delta

SHORTS=ROOT/'tmp/frame-execution/accepted-tail-mapped1'
RTL=ROOT/'tmp/frame-execution/accepted-tail-ofp3-rtl1'
EQUIV=ROOT/'tmp/frame-execution/accepted-tail-control-equivalence1'
SERIAL=ROOT/'tmp/frame-execution/accepted-tail-mapped1-serial/new'
TAIL=ROOT/'tmp/frame-execution/accepted-tail-shorts2'
FULL=ROOT/'tmp/frame-execution/accepted-tail-frame-full1'

def configure(module):
 for n in ('SHORTS','RTL','EQUIV','SERIAL'):setattr(module,n,globals()[n])

def parent_evidence(parent,verify):
 from finish_local_decode_release import parent_evidence as original
 return original(parent,verify)

def extra_evidence(board,batch,physical,serial,bind,verify):
 import finish_local_decode_release as original
 configure(original);original.extra_evidence(board,batch,physical,serial,bind,verify)
 assert audit_delta(board)==bind(board/'delta_audit.json')
 import audit_feedback_selection_map as selector
 selector.audit_delta=audit_delta
 assert selector.audit(board)==bind(board/'feedback_selection_map_audit.json')
 import accepted_tail_flow as flow
 import cached_head_flow as old
 flow.old_parent_proofs=old.proofs
 flow.short_proofs(board,TAIL)
 result=bind(TAIL/'result.json');meta=bind(TAIL/'inputs.json')
 verify({r['snapshot']:r['sha256'] for r in meta['inputs']})
 verify({str(TAIL/(n+'.sv')):h for n,h in meta['testbenches'].items()})
 verify({str(TAIL/(r['top']+'.log')):r['log_sha256'] for r in result['runs']})
 result=bind(FULL/'result.json');meta=bind(FULL/'inputs.json')
 assert result['status']=='REGISTERED_ACCEPTED_A_FULL_PASS' and result['inputs_sha256']==sha(FULL/'inputs.json')
 assert meta['board_inputs_sha256']==sha(board/'inputs.json') and result['package_sha256']==meta['package_sha256']
 from accepted_a_assets import accepted_package,PACKAGE_SHA
 accepted_package();assert result['package_sha256']==PACKAGE_SHA
 verify({r['snapshot']:r['sha256'] for r in meta['inputs']});verify(meta['vector_hashes'])
 verify({meta['binary']:meta['binary_sha256'],str(FULL/'simulation.log'):result['log_sha256']})
 driver=batch.parent/(batch.name+'-driver.py');dm=bind(driver.with_suffix('.json'))
 verify({str(driver):dm['driver_sha256'],dm['source']:dm['source_sha256'],str(HERE/'cached_head_flow.py'):dm['wrapper_sha256']})
 for n in ('accepted_tail_board.py','accepted_tail_flow.py','audit_accepted_a_tail.py','validate_accepted_tail.py',
           'validate_ownership_accepted_a.py','accepted_a_assets.py','run_accepted_tail_implementation.py','finish_accepted_tail_release.py'):
  verify({str(HERE/n):sha(HERE/n)})

def main():
 # Parent transform audits retain their imported original functions.
 import local_control_decode
 local_control_decode.audit_delta=audit_delta
 import finish_local_decode_release as original
 original.audit_delta=audit_delta;configure(original)
 package=original.package_code;execute=original.execute
 def new_package():
  source,code=package()
  before='from finish_local_decode_release import SHORTS,RTL,SERIAL,extra_evidence,parent_evidence'
  assert code.count(before)==1
  code=code.replace(before,'from finish_accepted_tail_release import SHORTS,RTL,SERIAL,extra_evidence,parent_evidence,FULL')
  code=code.replace('局部精确解码和OFP3独立DDR观察','定稿A尾项专用化、精确解码及OFP3独立DDR观察')
  before="functional=parent['functional'],"
  assert code.count(before)==1
  code=code.replace(before,"functional={**parent['functional'],'accepted_a_tail_full':dict(path=str(FULL),result_sha256=sha(FULL/'result.json'),whole_board=False,ddr_phy=False)},accepted_tail_profile=dict(allowed_coefficients=[0,4,8],unsupported_load_rejected=True,arbitrary_b1_weights_supported=False),")
  before='所有本版离线门禁和资源/时序绑定于manifest。'
  assert code.count(before)==1
  code=code.replace(before,'本实现仅接受已定稿A的第九项系数域0/4/8，装载其他尾项将报错停止；不支持任意替换B1权重。A包和执行程序、拓扑及计算周期未变。\n'+before)
  return source,code
 def new_execute(source,code,driver,args):
  code=code.replace("'finish_local_decode_release.py',['pll'","'finish_accepted_tail_release.py',['pll'")
  code=code.replace("'finish_local_decode_release.py',['bundle'","'finish_accepted_tail_release.py',['bundle'")
  return execute(source,code,driver,args)
 original.package_code=new_package;original.execute=new_execute;original.main()
if __name__=='__main__':main()
