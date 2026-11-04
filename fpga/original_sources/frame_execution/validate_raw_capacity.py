"""Full 1080p RAW roundtrip with the reduced 256-beat CDC queues.
Behavioral AXI service has random stalls and a periodic 200-cycle write pause;
this proves that profile, not an unlimited real DDR blackout guarantee.
"""
import argparse,json,re,shutil,traceback
from pathlib import Path
from run_units import HERE,ROOT,MS,sha,run
from map_connected_board import BASE

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--current-raw',action='store_true')
    p.add_argument('--audit-existing',action='store_true',help='Audit an already completed log without rerunning simulation')
    a=p.parse_args();root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp')
    root.mkdir(parents=True,exist_ok=a.audit_existing)
    names=['guarded_async_fifo.sv','raw32_frame_packer.sv','raw32_frame_cdc.sv',
        'axi_frame_write_dma.sv','camera_axi_write_path.sv','triple_frame_bank_manager.sv',
        'axi_frame_read_dma.sv','raw_ddr_frame_bridge.sv']
    manifest=json.loads((ROOT/'tmp/frame-execution/connected-board-map4/inputs.json').read_text())
    board={Path(x['snapshot']).name:Path(x['snapshot']) for x in manifest['inputs']}
    paths=[board[n] for n in names+['DC_FIFO.v']]+[HERE/'tb_raw_ddr_capacity.sv']
    if a.current_raw:
        changed={'guarded_async_fifo.sv','raw32_frame_cdc.sv','camera_axi_write_path.sv'}
        paths=[HERE.parent/'video_scaler'/f.name if f.name in changed else f for f in paths]
    if a.audit_existing:
        records=json.loads((root/'inputs.json').read_text())['inputs']
    else:
        records=[]
        for src in paths:
            dst=root/src.name;shutil.copy2(src,dst)
            records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
        (root/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__),
            camera_fifo_depth=256,read_fifo_depth=256,max_burst=128,frame_pixels=2073600),indent=2))
    try:
        if a.audit_existing:
            assert 'Errors: 0, Warnings: 0' in (root/'compile.log').read_text()
            body=(root/'simulation.log').read_text();seconds=None
            assert re.search(r'^# End time:.*Elapsed time:',body,re.M), 'Log not complete'
            assert not re.search(r'(?m)^#?\s*(FAIL|\*\* (Error|Fatal))',body)
        else:
            run([MS/'vlib.exe','work'],root,'vlib.log')
            body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log')
            assert 'Errors: 0, Warnings: 0' in body
            body,seconds=run([MS/'vsim.exe','-c','work.tb_raw_ddr_capacity','-do',
                'onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',timeout=600)
        lines=re.findall(r'^# (PASS raw DDR frame bridge:.*)$',body,re.M)
        assert len(lines)==1 and 'Errors: 0,' in body
        # The unchanged read-side FIFO intentionally does not consume the new
        # optional resident-credit output. Admit these exact elaboration notes
        # only; other warning kinds/ports still fail this regression.
        warnings=re.findall(r'^# \*\* Warning: (.*)$',body,re.M)
        for warning in warnings:
            assert ('raw_ddr_frame_bridge.sv(228)' in warning and
                    ("Missing connection for port 'rd_available'" in warning or
                     'Too few port connections. Expected 15, found 14.' in warning)), warning
        for r in records:assert sha(r['snapshot'])==r['sha256']
        result=dict(status='RAW_REDUCED_FIFO_FULL_FRAMES_FOURSTATE_PASS',lines=lines,seconds=seconds,
            inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),
            tested_good_frames=3,bad_frame_not_published=True,full_system=False,
            service_bound_for_real_shared_ddr_proven=False)
        result.update(audited_existing=a.audit_existing,auditor_sha256=sha(__file__),accepted_unused_output_notes=warnings)
        (root/('audit_result.json' if a.audit_existing else 'result.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    except Exception as exc:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2));raise

if __name__=='__main__':main()
