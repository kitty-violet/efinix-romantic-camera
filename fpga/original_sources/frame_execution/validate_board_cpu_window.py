"""Verify the actual board CPU program/development DDR window, four seeds."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import HERE,ROOT,MS,sha,run

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--rtl',type=Path);p.add_argument('--full-protocol',action='store_true');a=p.parse_args()
root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
board=HERE/'board_integration/ti60f225_oob_top.v'
text=board.read_text()
match=re.search(r'\.CPU_ADDRESS_BASE\(32\x27h([0-9a-f]+)\),\.CPU_ADDRESS_LIMIT\(32\x27h([0-9a-f]+)\),\.CPU_DDR_BASE\(32\x27h([0-9a-f]+)\)',text)
assert match
base,limit,ddr=(int(x,16) for x in match.groups())
assert (base,limit,ddr)==(0xe1000000,0xe1ffffff,0x04000000)
# Check the generated CPU's real internal decoder as well as our adapter.
cpu=ROOT/'tmp/sapphire-memory-ab/area15-i3-attempt2/rtl/EfxSapphireSoc.v'
assert "((io_input_cmd_payload_fragment_address & (~ 32'h00ffffff)) == 32'he1000000)" in cpu.read_text()
assert base==0xe1000000 and limit-base+1==0x1000000
for name in ('cnn_video_pipeline.sv','cnn_rgb_frame_pipeline.sv'):
    assert "CPU_ADDRESS_LIMIT=32'he1ffffff" in (HERE/name).read_text()
paths=[HERE.parent/'ddr_fabric'/n for n in ('sapphire_axi32_to_axi128.sv','tb_sapphire_axi32_to_axi128.sv')]
if a.rtl:
    paths[0]=a.rtl.resolve();assert paths[0].is_file()
records=[]
for src in paths:
    dst=root/src.name;shutil.copy2(src,dst);records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
(root/'inputs.json').write_text(json.dumps(dict(inputs=records,board_sha256=sha(board),
    cpu_base=base,cpu_inclusive_limit=limit,physical_base=ddr,runner_sha256=sha(__file__)),indent=2))
run([MS/'vlib.exe','work'],root,'vlib.log')
body,_=run([MS/'vlog.exe','-sv',*[r['snapshot'] for r in records]],root,'compile.log')
assert 'Errors: 0, Warnings: 0' in body
results=[]
for seed in (1,7,101,20260928):
    if a.full_protocol:
        log=f'seed{seed}-full-protocol.log'
        body,secs=run([MS/'vsim.exe','-c','work.tb_sapphire_axi32_to_axi128',
            '+SEED='+str(seed),'-do','onerror {quit -code 1}; run -all; quit -code 0'],root,log)
        lines=re.findall(r'^# (PASS SAPPHIRE_AXI32_128 seed=.*)$',body,re.M)
        assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
        results.append(dict(seed=seed,case='full_protocol',line=lines[0],seconds=secs,log_sha256=sha(root/log)))
    for bad in (0,1):
        args=[MS/'vsim.exe','-c','work.tb_sapphire_axi32_to_axi128','-gTRANSLATION=1',
            '-gBAD_MAPPING='+str(bad),'-gWINDOW_SOURCE_BASE='+str(base),
            '-gWINDOW_SOURCE_LIMIT='+str(limit),'-gWINDOW_DDR_BASE='+str(ddr),
            '+SEED='+str(seed),'-do','onerror {quit -code 1}; run -all; quit -code 0']
        log=f'seed{seed}-bad{bad}.log';body,secs=run(args,root,log)
        lines=re.findall(r'^# (PASS SAPPHIRE_AXI32_128_TRANSLATION .*)$',body,re.M)
        assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
        results.append(dict(seed=seed,bad_mapping=bad,line=lines[0],seconds=secs,log_sha256=sha(root/log)))
for record in records:assert sha(record['snapshot'])==record['sha256']
result=dict(status='BOARD_CPU_WINDOW_FOUR_SEED_PASS',results=results,inputs_sha256=sha(root/'inputs.json'),
    full_protocol=a.full_protocol,cpu_firmware_boot_tested=False,full_system=False)
(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
