"""Same bounded flows for the registered feedback-selection candidate."""
import sys
from rq_feedback_select_board import audit_delta

def bind():
    import cached_head_flow as old
    import local_control_decode
    old.audit_delta=audit_delta
    local_control_decode.audit_delta=audit_delta
    return old

def main():
    mode=sys.argv[1];old=bind()
    if mode=='serial':
        # Prior measured serial replay was 236.765s. Allow bounded CPU
        # contention with placement instead of timing out at 240s.
        import validate_control_ddr as control
        original=control.driver_run
        def serial_budget(source,code,root,args):
            assert code.count("'simulation.log',240)")==1
            return original(source,code.replace("'simulation.log',240)","'simulation.log',300)"),root,args)
        control.driver_run=serial_budget
    if mode=='rq':
        sys.argv.pop(1)
        import validate_rq_cached_head as check
        check.audit_delta=audit_delta;previous=check.whole_tb
        def tb():
            source,text=previous()
            text=text.replace('module tb_fixed_split_rq;',
                'module tb_fixed_split_rq;\n wire feedback_post_selected;')
            text=text.replace('#2;differential_cycles=differential_cycles+1;',
                '#2;differential_cycles=differential_cycles+1;\n'
                '  if(dut.issue && dut.chosen!=0 && feedback_post_selected!==in_ready[2])\n'
                '   $fatal(1,"RESERVED_KIND_DIFF");')
            return source,text
        check.whole_tb=tb;check.main();return
    if mode=='cluster':
        sys.argv.pop(1)
        import validate_cached_head_cluster as check
        check.audit_delta=audit_delta;check.main();return
    if mode=='equivalence':
        sys.argv.pop(1)
        import validate_local_control_decode as check
        check.audit_delta=audit_delta;check.main();return
    if mode=='head-audit':
        sys.argv.pop(1)
        import audit_cached_head_map as check
        from pathlib import Path
        import json
        check.audit_delta=audit_delta
        board=Path(sys.argv[1]).resolve();out=board/'cached_head_map_audit.json'
        assert not out.exists();result=check.audit(board)
        out.write_text(json.dumps(result,indent=2));print(result['status']);return
    old.main()

if __name__=='__main__':main()
