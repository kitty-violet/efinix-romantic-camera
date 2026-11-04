"""Whole-report targeted data replication plus exact late writer error factoring."""
import inspect,json
from pathlib import Path
from run_units import ROOT,HERE,sha
import native_local_drivers as replica
from native_local_drivers import graph

PARENT=ROOT/'tmp/frame-execution/connected-board-cnn-run-budget1'
OUT=ROOT/'tmp/frame-execution/cnn-timing-closure-native1'
WRITER=ROOT/'tmp/frame-execution/cnn-budget-writer-control-analysis1'
SHORT=ROOT/'tmp/frame-execution/cnn-budget-writer-control-short2'

def main():
    assert not OUT.exists()
    w=json.loads((WRITER/'result.json').read_text())
    for f,h in w['evidence'].items():assert sha(f)==h,f
    t=json.loads((SHORT/'result.json').read_text());assert t['status']=='NATIVE_WRITER_ERROR_COFACTOR_OFFICIAL_LUT_SHORT_PASS_NOT_MAPPED_RELEASE'
    assert sha(SHORT/'inputs.json')==t['inputs_sha256'] and sha(SHORT/'simulation.log')==t['simulation_sha256']
    surveyfile=ROOT/'tmp/board-test/cnn-budget-routed-survey-analysis1.json';s=json.loads(surveyfile.read_text())
    for f,h in s['evidence'].items():assert sha(f)==h,f
    code=inspect.getsource(replica.transform).replace("name=f'LOCAL_DRIVER_{idx:03d}';wire=f'local_driver_net_{idx:03d}'","name=f'CLOSE_DRIVER_{idx:03d}';wire=f'close_driver_net_{idx:03d}'")
    assert 'CLOSE_DRIVER_' in code
    # Hard local budget: at most512 added ordinary cells, never a resource fit claim.
    assert code.count('0<len(new)<=320')==1;code=code.replace('0<len(new)<=320','0<len(new)<=512')
    ns=dict(replica.__dict__);exec(compile(code,__file__+':targeted-clones','exec'),ns)
    placement=ROOT/'tmp/frame-execution/cnn-run-budget-convergence-bound1/outflow/ti60f225_oob.out.placeloc'
    source=(WRITER/'candidate.v').read_text()
    changed,rows,proof=ns['transform'](source,placement.read_text(),dict(paths=s['negative_paths']))
    assert not OUT.exists();OUT.mkdir()
    (OUT/'native_source.v').write_text(changed,encoding='utf8')
    (OUT/'replications.json').write_text(json.dumps(rows,indent=2))
    shutil=None
    (OUT/'writer_cofactor_proof.json').write_bytes((WRITER/'cofactor_proof.json').read_bytes())
    proof.update(status='CNN_TIMING_CLOSURE_JOINT_NATIVE_GRAPH_PASS',parent=str(PARENT),writer_luts=w['added_luts'],
        writer_ce_groups=w['ce_groups'],writer_ce_loads=w['changed_ce_loads'],writer_levels_before=w['old_max_error_levels'],writer_levels_after=1,
        writer_official_short_sha256=sha(SHORT/'result.json'),source_sha256=sha(OUT/'native_source.v'),replications_sha256=sha(OUT/'replications.json'),
        inputs={str(f):sha(f) for f in (Path(__file__),HERE/'native_local_drivers.py',WRITER/'result.json',WRITER/'candidate.v',WRITER/'cofactor_proof.json',
         SHORT/'result.json',SHORT/'inputs.json',SHORT/'simulation.log',surveyfile,placement,PARENT/'outflow/ti60f225_oob.map.v')},
        rejected_unplaced_new_factor_inputs=True,new_ff_only_existing_synchronous_process_clock=True,
        physical_pass=False,board_release=False)
    proof['total_added_kinds']=dict(proof['added_kinds']);proof['total_added_kinds']['EFX_LUT4']=proof['total_added_kinds'].get('EFX_LUT4',0)+w['added_luts']
    (OUT/'result.json').write_text(json.dumps(proof,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in proof.items() if k not in ('inputs',)}))

if __name__=='__main__':main()
