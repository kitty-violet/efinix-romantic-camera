"""Use the existing bounded flows with the explicitly checked transition delta."""
import json,sys
from pathlib import Path
from run_units import ROOT,sha
from complete_control_transition import audit_delta,BASE,REL,circuit

def proof(board,folder):
    audit_delta(board)
    r=json.loads((folder/'result.json').read_text());i=json.loads((folder/'inputs.json').read_text())
    assert r['status']=='COMPLETE_NEXT_STATE_90112_OFFICIAL_LUT_CHECKS_PASS'
    assert r['inputs_sha256']==sha(folder/'inputs.json')
    assert i['board_inputs_sha256']==sha(board/'inputs.json') and i['soc_sha256']==sha(board/REL)
    assert r['log_sha256']==sha(folder/'simulation.log')
    for n,h in i['files'].items():assert sha(folder/n)==h
    return dict(path=str(folder),result_sha256=sha(folder/'result.json'),
        unchanged_datapath_parent_inputs_sha256=sha(BASE/'inputs.json'))

def main():
    import cached_head_flow as flow
    flow.audit_delta=audit_delta;flow.BASE=BASE;flow.proofs=proof
    mode=sys.argv[1]
    if mode=='map':flow.main();return
    if mode=='equivalence':
        sys.argv.pop(1)
        import validate_local_control_decode as check
        old_decoder=check.decoder
        check.BASE=BASE;check.audit_delta=audit_delta
        check.decoder=lambda:old_decoder()+'\n'+circuit()[0]
        check.main()
        root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
        result=json.loads((root/'result.json').read_text())
        result['status']='COMPLETE_TRANSITION_CYCLE_EQUIVALENCE_PASS'
        result['adapter_sha256']=sha(__file__)
        result['reference_board']=str(BASE)
        (root/'result.json').write_text(json.dumps(result,indent=2))
        return
    if mode in ('rtl','startup','observe','serial'):
        flow.validate(mode,sys.argv[1:]);return
    raise ValueError(mode)
if __name__=='__main__':main()
