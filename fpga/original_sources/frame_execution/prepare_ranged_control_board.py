"""Re-evaluate proven range predecode under today's capacity policy; no new cycles."""
import argparse,json,re,shutil,subprocess,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from prepare_local_control_board import audit_delta as audit_parent,audit_branches
from audit_control_state_map import audit as audit_state
from audit_snapshot_cells import audit as audit_sync
from windows_process_tree import stop_launched_tree
BASE=ROOT/'tmp/frame-execution/connected-board-local-controls1'
RQ=HERE/'variants/fixed_bias_range_ready/fixed_rq32_mixed.sv'
PROOFS=[ROOT/'tmp/frame-execution'/n for n in ('bias-range-exhaustive1','bias-range-rq1','bias-range-rq-reg1')]

def check_proofs():
    audit_parent(BASE)
    text=RQ.read_text();restored=text
    lo=restored.index(' // Classify the stable header bias once per channel.')
    hi=restored.index(' wire [31:0] split_lane;',lo)
    restored=restored[:lo]+restored[hi:]
    lo=restored.index('  wire [12:0] range_class=');hi=restored.index('  wire fits7=',lo)
    restored=restored[:lo]+'  wire fits19=sum[31:19]=={13{sum[18]}};\n  wire fits25=sum[31:25]=={7{sum[24]}};\n'+restored[hi:]
    assert restored==(BASE/'assembly/fixed_rq32_mixed.sv').read_text(),'Unexpected historical/current delta'
    for proof in PROOFS:
        result=json.loads((proof/'result.json').read_text());meta=json.loads((proof/'inputs.json').read_text())
        assert result['status'] in ('BIAS_RANGE_READY_EXHAUSTIVE_PASS','FIXED_SPLIT_RQ_PASS')
        assert sha(proof/'inputs.json')==result['inputs_sha256'] and sha(proof/'simulation.log')==result['log_sha256']
        assert sha(proof/'fixed_rq32_mixed.sv')==sha(RQ)
        for rec in meta.get('inputs',[]):assert sha(rec['snapshot'])==rec['sha256']
    return {str(p):sha(p/'result.json') for p in PROOFS}

def audit_delta(board):
    changed=[]
    for rec in json.loads((board/'inputs.json').read_text())['inputs']:
        rel=Path(rec['snapshot']).relative_to(board);assert sha(board/rel)==rec['sha256']
        if sha(board/rel)!=sha(BASE/rel):changed.append(rel.as_posix())
    assert changed==['assembly/fixed_rq32_mixed.sv'],changed
    assert sha(board/changed[0])==sha(RQ)
    return dict(changed=changed,added_cycles=0,clock_pin_phy_reset_unchanged=True,hardware_access=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');p.add_argument('--map-prepared',action='store_true');a=p.parse_args()
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');proofs=check_proofs()
    if not a.map_prepared:
        assert not root.exists();root.mkdir()
        for d in ('rtl','assembly'):shutil.copytree(BASE/d,root/d)
        for f in BASE.iterdir():
            if f.is_file() and f.suffix in ('.xml','.sdc','.bin','.v','.sv','.vh','.svh'):shutil.copy2(f,root/f.name)
        shutil.copy2(RQ,root/'assembly/fixed_rq32_mixed.sv')
        meta=json.loads((BASE/'inputs.json').read_text())
        for rec in meta['inputs']:
            rel=Path(rec['snapshot']).relative_to(BASE);rec.update(snapshot=str(root/rel),sha256=sha(root/rel))
        meta.update(parent=str(BASE),parent_inputs_sha256=sha(BASE/'inputs.json'),range_proofs=proofs,range_runner_sha256=sha(__file__),board_download_allowed=False)
        (root/'inputs.json').write_text(json.dumps(meta,indent=2));shutil.copy2(__file__,root/'runner.py')
    (root/'delta_audit.json').write_text(json.dumps(audit_delta(root),indent=2))
    if a.prepare_only:print('RANGED_BOARD_PREPARED_NOT_RELEASE');return
    assert not (root/'map.log').exists()
    (root/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat ti60f225_oob.xml --flow map --timeout 900\nexit /b %errorlevel%\n')
    start=time.monotonic()
    try:
        with (root/'map.log').open('x') as log:
            child=subprocess.Popen(['cmd.exe','/d','/c','run.cmd'],cwd=root,stdout=log,stderr=subprocess.STDOUT)
            (root/'progress.json').write_text(json.dumps(dict(stage='map',pid=child.pid,host_budget_s=960)))
            try:code=child.wait(timeout=960)
            except subprocess.TimeoutExpired:
                (root/'timeout_cleanup.json').write_text(json.dumps(stop_launched_tree(child),indent=2));raise
        assert code==0 and re.search(r'map\s*:\s*PASS',(root/'map.log').read_text())
        net=root/'outflow/ti60f225_oob.map.v'
        audits={'local_branches_audit.json':audit_branches(net),'control_state_map_audit.json':audit_state(net,'cpu/','process_clk_150'),
          'snapshot_sync_audit.json':audit_sync(net.read_text(),prefix='cpu/system_hardJtag_debug_logic_jtagBridge/observation/',system='process_clk_150',tck='jtag_inst1_TCK')}
        for name,result in audits.items():(root/name).write_text(json.dumps(result,indent=2))
        ws=(root/'outflow/ti60f225_oob.warn.log').read_text(errors='replace');old=(BASE/'outflow/ti60f225_oob.warn.log').read_text(errors='replace');pattern=r'^\[EFX-0201 WARNING\] (.*?)\. \('
        assert not set(re.findall(pattern,ws,re.M))-set(re.findall(pattern,old,re.M)) and 'EFX-0654' not in ws
        audit_delta(root)
        result=dict(status='RANGE_LOCAL_CONTROL_FULL_MAP_NOT_RELEASE',seconds=time.monotonic()-start,inputs_sha256=sha(root/'inputs.json'),map_sha256=sha(net),
            report_sha256=sha(root/'outflow/ti60f225_oob.map.rpt'),audit_sha256={n:sha(root/n) for n in audits},snapshot_sync_audit_sha256=sha(root/'snapshot_sync_audit.json'),physical_pass=False,board_download_allowed=False)
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    except Exception as exc:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
