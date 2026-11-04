"""Read-only hardware evidence audit; all new simulations write to training workspace."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from .runtime import configure_workspace, contract_guard, file_hash, write_json
from .fpga_video_resize import verify as resize_verify


ROOT=Path('E:/fpga')
HERE=ROOT/'hw/efinity/prototypes/frame_execution'
SCALER=ROOT/'hw/efinity/prototypes/video_scaler'
MS=Path('E:/altera/15.0/modelsim_ase/win32aloem')


def run(args,cwd,name,env=None,timeout=120):
    begin=time.monotonic()
    with (cwd/name).open('w',encoding='utf-8') as f:
        result=subprocess.run([str(a) for a in args],cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,
                              timeout=timeout,creationflags=subprocess.CREATE_NO_WINDOW)
    body=(cwd/name).read_text(encoding='utf-8',errors='replace')
    if result.returncode or re.search(r'\bFAIL\b|TEST_FAIL|%Error|\*\* (Error|Fatal)',body):
        raise RuntimeError(f'{name}: {body[-2000:]}')
    return body,time.monotonic()-begin


def main():
    root=configure_workspace();out=root/'runs/sampling-hardware-audit-20261003'
    out.mkdir(exist_ok=False);contract_guard()
    selection_path=HERE/'board_resource_candidate.json';selection=json.loads(selection_path.read_text())
    board=ROOT/selection['selected_board']
    manifest=json.loads((board/'inputs.json').read_text())
    assert file_hash(board/'inputs.json')==selection['selected_inputs']['sha256']
    assert file_hash(board/'result.json')==selection['selected_result']['sha256']
    resource=json.loads((board/'resource_audit3.json').read_text())
    assert file_hash(board/'inputs.json')==resource['inputs_sha256']
    assert file_hash(board/'outflow/ti60f225_oob.map.rpt')==resource['report_sha256']
    source={Path(r['snapshot']).name:r for r in manifest['inputs']}
    records=[]
    for record in manifest['inputs']:
        assert file_hash(record['snapshot'])==record['sha256'],record['snapshot']
        records.append({'path':record['snapshot'],'sha256':record['sha256']})
    src=out/'rtl';src.mkdir()
    names=['rgb_area_downscaler_3to2.sv','rgb_bilinear_scaler_prototype.sv',
           'resize_axis_down_3_to_2.sv','resize_axis_up_2_to_3.sv']
    for name in names:shutil.copy2(source[name]['snapshot'],src/name)
    # Arithmetic helpers are isolated reference probes; full scaler below uses its own RTL.
    helper_names=['lerp_u8_denom4.sv','lerp_u8_denom6.sv']
    tests=['tb_rgb_area_downscaler_3to2','tb_rgb_bilinear_scaler','tb_resize_axis','tb_lerp_u8']
    for name in helper_names+[t+'.sv' for t in tests]:shutil.copy2(SCALER/name,src/name)
    # Strengthen the old constant full-frame test with nonconstant RGB patterns.
    body=(SCALER/'tb_rgb_area_downscaler_full_frame.sv').read_text()
    body=body.replace('tb_rgb_area_downscaler_full_frame','tb_area_pattern_full')
    funcs='''
    function automatic [7:0] sample(input integer x,y,c);
      case(c)
        0: sample=(29*x+17*y+(x*y)%251)&255;
        1: sample=((x^y)*37+3*y)&255;
        default: sample=(73*(x%4)+89*(y%3)+11)&255;
      endcase
    endfunction
    function automatic [23:0] pixel(input integer x,y);
      pixel={sample(x,y,0),sample(x,y,1),sample(x,y,2)};
    endfunction
    function automatic [7:0] expected_channel(input integer x,y,c);
      integer sx,sy,wx0,wx1,wy0,wy1;
      begin
        sx=3*(x/2)+(x%2);sy=3*(y/2)+(y%2);
        wx0=(x%2)==0?2:1;wx1=3-wx0;wy0=(y%2)==0?2:1;wy1=3-wy0;
        expected_channel=(sample(sx,sy,c)*wx0*wy0+sample(sx+1,sy,c)*wx1*wy0+
                          sample(sx,sy+1,c)*wx0*wy1+sample(sx+1,sy+1,c)*wx1*wy1+4)/9;
      end
    endfunction
    function automatic [23:0] expected(input integer index);
      expected={expected_channel(index%DW,index/DW,0),expected_channel(index%DW,index/DW,1),expected_channel(index%DW,index/DW,2)};
    endfunction
'''
    body=body.replace('    always #3.333',funcs+'\n    always #3.333')
    body=body.replace("m_rgb !== 24'h345678",'m_rgb !== expected(output_count)')
    body=body.replace("s_valid <= 1'b1;","s_valid <= 1'b1;\n            s_rgb <= pixel(input_count%SW,input_count/SW);")
    body=body.replace('PASS AREA_FULL','PASS AREA_PATTERN_FULL')
    (src/'tb_area_pattern_full.sv').write_text(body)
    tests.append('tb_area_pattern_full')
    write_json(out/'inputs.json',{'selected_board':str(board),'selection_sha256':file_hash(selection_path),
        'board_inputs_sha256':file_hash(board/'inputs.json'),'all_frozen_sources':records,
        'simulation_sources':{str(p):file_hash(p) for p in sorted(src.iterdir())},
        'runner_sha256':file_hash(Path(__file__))})
    shutil.copy2(Path(__file__),out/Path(__file__).name)
    write_json(out/'resource_snapshot.json',{'selected_board':str(board),'official_resources':resource['official_resources'],
        'ram_blocks':resource['ram_blocks'],'physical_dsp':resource['physical_dsp'],'device':resource['device'],
        'result':json.loads((board/'result.json').read_text()),'selection':selection})
    sim=out/'modelsim';sim.mkdir()
    run([MS/'vlib.exe','work'],sim,'vlib.log')
    body,secs=run([MS/'vlog.exe','-sv',*[str(p) for p in sorted(src.glob('*.sv'))]],sim,'compile.log')
    assert 'Errors: 0' in body
    results=[]
    for top in tests:
        body,secs=run([MS/'vsim.exe','-c',top,'-do','onerror {quit -code 1}; run -all; quit -code 0'],sim,top+'.log')
        lines=[line for line in body.splitlines() if re.search(r'\bPASS\b|TEST_PASS',line)]
        assert lines,(top,body[-2000:])
        results.append({'test':top,'pass_lines':lines,'seconds':secs,'log_sha256':file_hash(sim/(top+'.log'))})
        print('RTL_PASS',top,round(secs,3),lines,flush=True)
        write_json(out/'progress.json',{'completed':results,'status':'RUNNING'})
    # Replay the exact previously built numeric kernel in this workspace, after source/binary binding.
    prior_dir=ROOT/'tmp/frame-execution/paired-writer-cluster3'
    native=json.loads((prior_dir/'native_inputs.json').read_text());frozen=Path(native['snapshot'])
    assert file_hash(native['binary'])==native['binary_sha256']
    assert file_hash(frozen/'inputs.json')==native['source_manifest_sha256']
    ns=json.loads((frozen/'inputs.json').read_text());overlap=[]
    for r in ns['inputs']:
        assert file_hash(r['snapshot'])==r['sha256']
        name=Path(r['snapshot']).name
        if name in source:
            assert r['sha256']==source[name]['sha256'],name
            overlap.append(name)
    kernel=out/'kernel';kernel.mkdir()
    exe=kernel/Path(native['binary']).name;shutil.copy2(native['binary'],exe)
    env=os.environ.copy();env['PATH']=str(ROOT/'tools/verilator/ucrt64/bin')+';E:/GIT/Git/usr/bin;'+env['PATH']
    env.update(TEMP=str(kernel),TMP=str(kernel),PYTHONDONTWRITEBYTECODE='1')
    verify=subprocess.list2cmdline([sys.executable,'-B',str(HERE.parent/'coefficient_store/validate_readback.py'),
                                  '--image-dir',str(frozen/'vectors'),'--readback',str(kernel/'readback.hex')])
    body,secs=run([exe,'+VECTORS='+(frozen/'vectors').as_posix(),'+DUMP='+(kernel/'readback.hex').as_posix(),
                   '+VERIFY_COMMAND='+verify],kernel,'replay.log',env)
    lines=re.findall(r'^PASS COMPUTE_CLUSTER.*$',body,re.M)
    assert len(lines)==1 and 'ACTUAL_RTL_READBACK_SOFTWARE_VERIFIED' in body
    py=resize_verify()
    final={'status':'SAMPLING_CHECKS_PASS_WITH_SCOPED_LIMITS','rtl_tests':results,
           'kernel_replay':{'pass_line':lines[0],'seconds':secs,'matched_board_sources':overlap,
                            'binary_sha256':native['binary_sha256'],'log_sha256':file_hash(kernel/'replay.log'),
                            'four_state':False,'full_frame':False},
           'training_resize_reference':py,'selected_board':str(board),
           'scope':'Scaler four-state RTL, native kernel numeric tests; no physical timing, full-system release or trained INT8 deployment claim.',
           'shared_contract_sha256':file_hash(ROOT/'network_ab/model_b_contract.json'),
           'inputs_sha256':file_hash(out/'inputs.json')}
    for r in records:assert file_hash(r['path'])==r['sha256']
    write_json(out/'result.json',final);print(json.dumps(final,indent=2),flush=True)


if __name__=='__main__':main()
