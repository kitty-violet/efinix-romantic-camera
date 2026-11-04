"""Full existing release gates plus registered-feedback-selector map evidence."""
from pathlib import Path
from run_units import ROOT,HERE,sha
from rq_feedback_select_board import audit_delta
from feedback_select_flow import bind as bind_source

SHORTS=ROOT/'tmp/frame-execution/feedback-select-mapped1'
RTL=ROOT/'tmp/frame-execution/feedback-select-ofp3-rtl1'
EQUIV=ROOT/'tmp/frame-execution/feedback-select-control-equivalence1'
SERIAL=ROOT/'tmp/frame-execution/feedback-select-mapped1-serial/new'
RQ_SHORT=ROOT/'tmp/frame-execution/feedback-select-shorts1'
CLUSTER=ROOT/'tmp/frame-execution/feedback-select-cluster1'

def configure(module):
    for name in ('SHORTS','RTL','EQUIV','SERIAL','RQ_SHORT','CLUSTER'):
        setattr(module,name,globals()[name])

def parent_evidence(parent,verify):
    from finish_local_decode_release import parent_evidence as original
    return original(parent,verify)

def extra_evidence(board,batch,physical,serial,bind,verify):
    import finish_cached_head_release as cached
    import audit_cached_head_map as heads
    configure(cached);heads.audit_delta=audit_delta
    cached.extra_evidence(board,batch,physical,serial,bind,verify)
    from audit_feedback_selection_map import audit
    assert audit(board)==bind(board/'feedback_selection_map_audit.json')
    # Keep exact identity of the additional adapter and all three RTL changes.
    for name in ('rq_feedback_select_board.py','feedback_select_flow.py','audit_feedback_selection_map.py',
                 'finish_feedback_select_release.py','review_cached_head_place.py'):
        verify({str(HERE/name):sha(HERE/name)})

def main():
    bind_source()
    import finish_local_decode_release as original
    original.audit_delta=audit_delta;configure(original)
    package=original.package_code;execute=original.execute
    def new_package():
        source,code=package()
        code=code.replace('from finish_local_decode_release import SHORTS,RTL,SERIAL,extra_evidence,parent_evidence',
                          'from finish_feedback_select_release import SHORTS,RTL,SERIAL,extra_evidence,parent_evidence')
        code=code.replace('局部精确解码和OFP3独立DDR观察',
                          'RQ队首缓存、寄存任务类型反馈选择及OFP3独立DDR观察')
        return source,code
    def new_execute(source,code,driver,args):
        code=code.replace("'finish_local_decode_release.py',['pll'","'finish_feedback_select_release.py',['pll'")
        code=code.replace("'finish_local_decode_release.py',['bundle'","'finish_feedback_select_release.py',['bundle'")
        return execute(source,code,driver,args)
    original.package_code=new_package;original.execute=new_execute;original.main()

if __name__=='__main__':main()
