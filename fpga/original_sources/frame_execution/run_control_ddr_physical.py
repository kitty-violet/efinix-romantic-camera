"""Bounded complete-board gates, using selected seed 3 and original constraints."""
import argparse,json,sys,runpy,re,xml.etree.ElementTree as ET
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_ddr_combined import audit_delta,once

def audit_comb4(board):
    audit_delta(board)
    body=(board/'outflow/ti60f225_oob.map.v').read_text()
    assert not re.search(r'^\s*EFX_COMB4\s',body,re.M),'COMB4 remains in actual emitted map'
    # The synthesis header omits default-zero options; check the bound XML
    # and actual emitted cells, not presence of an optional log line.
    values=[e.get('value') for e in ET.parse(board/'ti60f225_oob.xml').iter() if e.get('name')=='pack-luts-to-comb4']
    assert values==['0']
    from audit_snapshot_cells import cells
    cc=cells(body);pairs=[]
    for i in range(2):
        first=f'cpu/shift_meta[{i}]~FF';second=f'cpu/shift_sync[{i}]~FF'
        for n in (first,second):
            k,p,a=cc[n];assert k=='EFX_FF' and p['CLK']=='process_clk_150' and 'async_reg=' in a
            assert p['CE']=="1'b1" and p['SR']=="1'b0"
        assert cc[second][1]['D']==cc[first][1]['Q']
        pairs.append([first,second])
    return dict(status='NO_COMB4_ACTUAL_MAP_AND_DDR_EVENT_SYNC_PASS',comb4_instances=0,event_sync_pairs=pairs,
        map_sha256=sha(board/'outflow/ti60f225_oob.map.v'),inputs_sha256=sha(board/'inputs.json'),hardware_access=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--reuse-shorts-from',type=Path);p.add_argument('--seed',type=int,default=3);a=p.parse_args()
    board=a.board.resolve();root=a.root.resolve();checked=audit_comb4(board)
    from audit_command_toggle_sync import audit as command_sync
    net=board/'outflow/ti60f225_oob.map.v';sync=command_sync(net.read_text());assert not sync['errors']
    for name,data in [('command_sync_audit.json',dict(sync,netlist=str(net),sha256=sha(net))),('combined_map_audit.json',checked)]:
        path=board/name
        if path.exists():assert json.loads(path.read_text())==data
        else:
            with path.open('x') as f:json.dump(data,f,indent=2)
    source=HERE/'run_control_command_physical.py';code=source.read_text()
    code=once(code,'from control_command_repair import audit_delta','from control_ddr_combined import audit_delta')
    code=once(code,"'validate_command_progress_mapped.py',['--board'","'validate_control_ddr.py',['startup','--board'")
    code=once(code,"'validate_command_progress_serial.py',['--startup'","'validate_control_ddr.py',['serial','--board',board,'--startup'")
    assert 1<=a.seed<=100
    code=once(code,"'--seed',1,",f"'--seed',{a.seed},")
    code=once(code,"'probe_shift_guard_physical.py'","'probe_control_ddr_physical.py'")
    # Give the same route algorithm a finite 25-minute budget; no broad retuning.
    code=once(code,"  import audit_control_state_implementation as impl",
        "  physical_result=json.loads((physical/'result.json').read_text())\n  assert all(s['exit_code']==0 for s in physical_result['stages']),physical_result\n  import audit_control_state_implementation as impl")
    code=once(code,"'--root',serial],240)","'--root',serial],300)")
    if a.reuse_shorts_from:
        previous=a.reuse_shorts_from.resolve();startup=previous/'startup';serial=previous.parent/(previous.name+'-serial')
        reused={}
        for folder in (startup,serial/'new'):
            result=json.loads((folder/'result.json').read_text());inputs=json.loads((folder/'inputs.json').read_text())
            assert result['inputs_sha256']==sha(folder/'inputs.json')
            assert inputs.get('source_map_sha256',inputs.get('original_map_sha256'))==sha(net)
            for name,digest in inputs['files'].items():assert sha(folder/name)==digest
            assert sha(folder/'simulation.log')==result.get('log_sha256',result.get('simulation_sha256'))
            reused[str(folder/'result.json')]=sha(folder/'result.json')
        check=json.loads((serial/'progress_counter_checks.json').read_text())
        assert check['status']=='CONTROL_DDR_MAPPED_SERIAL_IDLE_RESET_PASS'
        assert check['source_result_sha256']==sha(serial/'new/result.json')
        before=code.index("  run('startup'");after=code.index('  reference=',before)
        code=code[:before]+('  import shutil\n  shutil.copytree(Path('+repr(str(startup))+"),root/'startup')\n  serial=Path("+repr(str(serial))+")\n  (root/'short_evidence_reuse.json').write_text(json.dumps("+repr(reused)+",indent=2))\n")+code[after:]
    driver=root.parent/(root.name+'-combined-driver.py');assert not driver.exists() and not root.exists()
    compile(code,str(driver),'exec');driver.write_text(code)
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
        generated_sha256=sha(driver),wrapper_sha256=sha(__file__),mapped_workaround_audit=checked,actual_seed=a.seed),indent=2))
    sys.argv=[str(driver),'--board',str(board),'--root',str(root)];runpy.run_path(str(driver),run_name='__main__')
if __name__=='__main__':main()
