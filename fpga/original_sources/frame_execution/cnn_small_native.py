"""Two bounded candidates: writer factoring, optionally 64 pure-data clones."""
import inspect,json,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
import native_local_drivers as replica
from native_local_drivers import audit,graph
from complete_native_logic_split import protected_blocks
PARENT=ROOT/'tmp/frame-execution/connected-board-cnn-run-budget1'
WRITER=ROOT/'tmp/frame-execution/cnn-budget-writer-control-analysis1'
SHORT=ROOT/'tmp/frame-execution/cnn-budget-writer-control-short2'

def build(variant):
    assert variant in ('minimal','data64')
    out=ROOT/f'tmp/frame-execution/cnn-{variant}-native1';assert not out.exists()
    w=json.loads((WRITER/'result.json').read_text());t=json.loads((SHORT/'result.json').read_text())
    for f,h in w['evidence'].items():assert sha(f)==h
    assert t['status']=='NATIVE_WRITER_ERROR_COFACTOR_OFFICIAL_LUT_SHORT_PASS_NOT_MAPPED_RELEASE'
    assert sha(SHORT/'inputs.json')==t['inputs_sha256'] and sha(SHORT/'simulation.log')==t['simulation_sha256']
    selection=ROOT/'tmp/board-test/cnn-data-only-selection1.json';s=json.loads(selection.read_text())
    for f,h in s['inputs'].items():assert sha(f)==h
    surveyfile=ROOT/'tmp/board-test/cnn-budget-routed-survey-analysis1.json';sv=json.loads(surveyfile.read_text())
    for f,h in sv['evidence'].items():assert sha(f)==h
    source=(WRITER/'candidate.v').read_text();place=ROOT/'tmp/frame-execution/cnn-run-budget-convergence-bound1/outflow/ti60f225_oob.out.placeloc'
    if variant=='minimal':
        changed=source;rows=[];proof=audit(source,changed,rows)
        proof.update(selected_nets=0,changed_load_cells=0,max_loads_per_new_driver=0,covered_negative_endpoints=0)
    else:
        selected={r['net'] for r in s['selected']};assert len(selected)==14 and s['clones']==64
        code=inspect.getsource(replica.transform)
        old="name=f'LOCAL_DRIVER_{idx:03d}';wire=f'local_driver_net_{idx:03d}'"
        assert old in code;code=code.replace(old,"name=f'SMALL_DRIVER_{idx:03d}';wire=f'small_driver_net_{idx:03d}'")
        needle='loads=collections.defaultdict(list);unsafe=set()';assert needle in code
        code=code.replace(needle,'candidates={k:v for k,v in candidates.items() if k in SELECTED}\n    '+needle)
        env=dict(replica.__dict__);env['SELECTED']=selected
        exec(compile(code,__file__+':data-only','exec'),env)
        changed,rows,proof=env['transform'](source,place.read_text(),dict(paths=sv['negative_paths']))
        assert proof['clones']==64 and {r['net'] for r in rows}==selected
    # Shared CE, memory address and clock/reset ports must remain exactly wired.
    a,b=graph(source),graph(changed);allowed={'EFX_LUT4':{'I0','I1','I2','I3'},'EFX_FF':{'D'},'EFX_ADD':{'I0','I1','CI'}}
    changes=[]
    for n,(kind,ports) in a.items():
        for p,ts in ports.items():
            if b[n][1][p]!=ts:
                assert p in allowed.get(kind,set()),(n,kind,p)
                changes.append([n,p])
    assert protected_blocks(changed)==protected_blocks((PARENT/'outflow/ti60f225_oob.map.v').read_text())
    out.mkdir();(out/'native_source.v').write_text(changed,encoding='utf-8')
    (out/'replications.json').write_text(json.dumps(rows,indent=2))
    (out/'writer_cofactor_proof.json').write_bytes((WRITER/'cofactor_proof.json').read_bytes())
    proof.update(status='CNN_TIMING_SMALL_NATIVE_GRAPH_PASS',variant=variant,parent=str(PARENT),writer_luts=w['added_luts'],writer_ce_groups=w['ce_groups'],writer_ce_loads=w['changed_ce_loads'],writer_levels_before=w['old_max_error_levels'],writer_levels_after=1,
      shared_control_address_unchanged_from_writer=True,changed_data_ports=changes,
      source_sha256=sha(out/'native_source.v'),replications_sha256=sha(out/'replications.json'),
      inputs={str(p):sha(p) for p in (Path(__file__),HERE/'native_local_drivers.py',HERE/'cnn_small_board_factory.py',HERE/'cnn_timing_closure_board.py',HERE/f'cnn_{variant}_board.py',selection,surveyfile,place,WRITER/'result.json',WRITER/'candidate.v',WRITER/'cofactor_proof.json',SHORT/'result.json',SHORT/'inputs.json',SHORT/'simulation.log',PARENT/'outflow/ti60f225_oob.map.v')},
      new_ff_only_existing_synchronous_process_clock=True,physical_pass=False,board_release=False)
    proof['total_added_kinds']=dict(proof['added_kinds']);proof['total_added_kinds']['EFX_LUT4']=proof['total_added_kinds'].get('EFX_LUT4',0)+156
    (out/'result.json').write_text(json.dumps(proof,indent=2))
    print(variant,proof['status'],proof['total_added_kinds'],len(changes),flush=True)

if __name__=='__main__':build(sys.argv[1])
