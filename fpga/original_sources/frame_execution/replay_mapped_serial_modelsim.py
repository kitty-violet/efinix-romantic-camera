"""Replay identical unencrypted mapped primitives/TB on installed ModelSim.

No change to stimulus, delay, netlist, assertions, or physical design.
"""
import argparse,json,re,shutil,subprocess,time,traceback
from pathlib import Path
from run_units import ROOT,MS,sha
from read_control_ddr_progress import decode
from windows_process_tree import stop_launched_tree

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    source=a.source.resolve();root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    inputs=json.loads((source/'inputs.json').read_text());root.mkdir()
    for name,h in inputs['files'].items():
        path=source/name;assert sha(path)==h
        assert 'pragma protect' not in path.read_text(errors='replace')
        shutil.copy2(path,root/name)
    meta=dict(inputs,source_replay=str(source),source_inputs_sha256=sha(source/'inputs.json'),
        runner_sha256=sha(__file__),simulator=str(MS),stimulus_and_models_unchanged=True,hardware_access=False)
    (root/'inputs.json').write_text(json.dumps(meta,indent=2));(root/'runner.py').write_bytes(Path(__file__).read_bytes())
    start=time.monotonic()
    try:
        for args,name,budget in [(['vlib.exe','work'],'vlib.log',30),
            (['vlog.exe','-sv','-timescale','1ns/1ps',*[n for n in inputs['files'] if n.endswith('.v')],'tb.sv'],'compile.log',45),
            (['vsim.exe','-c','work.tb','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],'simulation.log',240)]:
            with (root/name).open('x') as f:
                proc=subprocess.Popen([str(MS/args[0]),*args[1:]],cwd=root,stdout=f,stderr=subprocess.STDOUT)
                try:code=proc.wait(timeout=budget)
                except subprocess.TimeoutExpired:
                    (root/'timeout_cleanup.json').write_text(json.dumps(stop_launched_tree(proc),indent=2));raise
            body=(root/name).read_text(errors='replace')
            assert code==0 and not re.search(r'(?m)^#?\s*\*\* (Error|Fatal)',body),(name,body[-1500:])
        assert 'PASS MAPPED_SERIAL_APB_OFS_RESET commands=4' in body
        records=[decode(int(v,16)) for v in re.findall(r'OFSRAW ([0-9a-fA-F]+)',body)]
        assert [r['live']['accepted_commands'] for r in records]==[1,2,4]
        assert [r['live']['replies'] for r in records]==[1,2,4]
        result=dict(status='MAPPED_SERIAL_APB_OFS_RESET_PASS',seconds=time.monotonic()-start,snapshots=records,
            inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),hardware_access=False,
            identical_stimulus_and_unencrypted_official_models=True)
        for name,h in inputs['files'].items():assert sha(root/name)==h
        (root/'result.json').write_text(json.dumps(result,indent=2))
        (root/'progress_counter_checks.json').write_text(json.dumps(dict(status='CONTROL_DDR_MAPPED_SERIAL_IDLE_RESET_PASS',
            accepted=[1,2,4],replies=[1,2,4],quiet_checks=3,quiet_cycles_each=4096,
            source_result_sha256=sha(root/'result.json'),source_map_sha256=inputs['source_map_sha256'],hardware_access=False),indent=2))
        (root/'paired_reply_result.json').write_text(json.dumps(dict(status='MAPPED_PAIRED_REPLY_STARTUP_FAULT_RESET_PASS',
            source_map_sha256=inputs['source_map_sha256'],result_sha256=sha(root/'result.json'),hardware_access=False),indent=2))
        print(json.dumps({k:result[k] for k in ('status','seconds','identical_stimulus_and_unencrypted_official_models')}))
    except Exception as exc:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
