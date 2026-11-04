"""Bind an observation-only board delta and verify real two-FF implementation."""
import argparse,json,re
from pathlib import Path
from run_units import ROOT,sha


def once(text, old, new=''):
    assert text.count(old)==1, ('Unexpected delta',old)
    return text.replace(old,new)


def audit(board):
    board=board.resolve();assert board.is_relative_to(ROOT/'tmp')
    base=ROOT/'tmp/frame-execution/connected-board-skip-credit1'
    meta=json.loads((board/'inputs.json').read_text())
    mapped=json.loads((board/'result.json').read_text())
    assert mapped['inputs_sha256']==sha(board/'inputs.json')
    changed=[]
    for row in meta['inputs']:
        path=Path(row['snapshot']);assert sha(path)==row['sha256']
        old=base/path.relative_to(board)
        if sha(old)!=sha(path):changed.append(path.relative_to(board).as_posix())
    expected=['rtl/ti60f225_oob_top.v','assembly/cnn_apb_control.sv',
              'assembly/cnn_video_pipeline.sv','assembly/cnn_rgb_frame_pipeline.sv']
    assert set(changed)==set(expected)
    for name in expected:
        old=(base/name).read_text();new=(board/name).read_text()
        if name.endswith('ti60f225_oob_top.v'):
            start=new.index('// Observation only: read twice after settling.')
            end=new.index('wire [31:0] committed_frames',start)
            block=new[start:end]
            assert 'syn_srlstyle="registers", syn_preserve="true"' in block
            assert 'bringup_ddr_sync<=bringup_ddr_meta;' in block
            new=new[:start]+new[end:]
            new=once(new,' .bringup_pipeline_status(bringup_pipeline_status),.bringup_ddr_status(bringup_ddr_sync),\n')
            new=once(new,'.rdlvl_shift                   (bringup_cal_phase),\n')
            new=once(new,'    .bringup_pipeline_status(bringup_pipeline_status),.pipeline_error', '    .pipeline_error')
        elif name.endswith('cnn_apb_control.sv'):
            new=once(new,' input wire [31:0] bringup_pipeline_status,bringup_ddr_status,\n')
            new=once(new,'   // Read-only observations; no control-path feedback.\n'
                     "   16'h000c:prdata=bringup_pipeline_status;\n   16'h006c:prdata=bringup_ddr_status;\n")
        elif name.endswith('cnn_video_pipeline.sv'):
            new=once(new,'    output wire [31:0] bringup_pipeline_status,\n')
            new=once(new,'        .bringup_pipeline_status(bringup_pipeline_status),.error', '        .error')
        else:
            new=once(new,' output wire [31:0] bringup_pipeline_status,\n')
            start=new.index(' wire [7:0] bringup_datapath_fault;')
            end=new.index(' assign error=owner_error',start)
            new=new[:start]+new[end:]
            for oldport in ('fault_code(bringup_datapath_fault)','program_fault_code(bringup_program_fault)',
                            'arbiter_protocol_error(bringup_arbiter_error)','cpu_protocol_error(bringup_cpu_error)'):
                new=once(new,oldport,oldport.split('(')[0]+'()')
        assert new.strip()==old.strip(),('Functional difference outside reviewed observation',name)
    def warnings(p):
        return set(re.findall(r'^\[EFX-0201 WARNING\] (.*?)\. \(',p.read_text(),re.M))
    w=board/'outflow/ti60f225_oob.warn.log'
    added=sorted(warnings(w)-warnings(base/'outflow/ti60f225_oob.warn.log'))
    assert not added and not mapped['comb_loop_warning']
    net=board/'outflow/ti60f225_oob.map.v';assert sha(net)==mapped['map_sha256']
    contents=net.read_text()
    cells=re.findall(r'EFX_FF\s+(\\bringup_ddr_(?:meta|sync)\[\d+\]~FF)\s+\((.*?)\) /\*',contents,re.S)
    assert len(cells)==64,('Expected 64 real diagnostic FFs',len(cells))
    ports=dict(cells)
    for k in range(32):
        first=ports[f'\\bringup_ddr_meta[{k}]~FF'];second=ports[f'\\bringup_ddr_sync[{k}]~FF']
        assert f'.Q(bringup_ddr_meta[{k}])' in first
        assert f'.D(bringup_ddr_meta[{k}])' in second and f'.Q(bringup_ddr_sync[{k}])' in second
        assert '.CLK(process_clk_150)' in first and '.CLK(process_clk_150)' in second
    return dict(status='OBSERVATION_ONLY_DELTA_AND_64_FF_PASS',base=str(base),board=str(board),
                changed_sources={name:sha(board/name) for name in expected},
                inputs_sha256=sha(board/'inputs.json'),map_sha256=sha(net),warning_sha256=sha(w),
                no_new_undriven=True,diagnostic_ff=64,functional_paths_reset_pins_clocks_unchanged=True,
                physical_timing_pass=False,board_release=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);a=p.parse_args()
    result=audit(a.board);out=a.board/'observation_audit.json';assert not out.exists()
    out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
