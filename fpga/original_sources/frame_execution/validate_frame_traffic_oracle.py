"""Short four-state gate for the exact traffic oracle used by full-frame TB."""
import argparse,json,re
from pathlib import Path
from run_units import ROOT,HERE,MS,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--descriptors',type=Path,required=True);a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
d=a.descriptors.resolve();meta=json.loads((d/'result.json').read_text())
raw=(d/'dma.bin').read_bytes();assert sha(d/'dma.bin')==meta['files']['dma.bin']
reads=(len((d/'compute.bin').read_bytes())+len(raw))//16;writes=responses=0
for start in range(0,len(raw),16):
    word=int.from_bytes(raw[start:start+16],'little');version=word>>124;assert version in (2,3)
    rgb=(word>>45)&1;store=(word>>14)&1;columns=(word>>20)&511;address=(word>>46)&((1<<28)-1)
    size=columns*(3 if rgb else 32);align=64 if version==3 and not rgb else 16
    # Enumerate physical 16-byte addresses independently of the RTL counters.
    beats=list(range(address//align*align,(address+size+align-1)//align*align,16))
    if not store:reads+=len(beats);continue
    writes+=len(beats);last=None;length=0
    for address in beats:
        if last is None or address//4096!=last//4096 or length==128:responses+=1;length=0
        last=address;length+=1
body=(HERE/'variants/row_pair_arena/tb_fourway_full.sv').read_text()
task=body[body.index(' task expected_traffic;'):body.index(' endtask',body.index(' task expected_traffic;'))+len(' endtask')]
tb='module tb; reg [7:0] program_bytes[0:729535]; integer expected_readbeats,expected_writebeats,expected_bresponses; integer fd,n;\n'
tb+='task fail(input string why);$fatal(1,"%s",why);endtask\n'+task+'\ninitial begin\n'
tb+='fd=$fopen("'+(d/'dma.bin').as_posix()+'","rb");n=$fread(program_bytes,fd,562496,167040);$fclose(fd);if(n!=167040)$fatal(1,"size");\nexpected_traffic();\n'
tb+=f'if(expected_readbeats!={reads} || expected_writebeats!={writes} || expected_bresponses!={responses})$fatal(1,"independent ledger mismatch");\n'
tb+='$display("PASS FRAME_TRAFFIC_ORACLE");$finish;end endmodule\n'
(r/'tb.sv').write_text(tb);run([MS/'vlib.exe','work'],r,'vlib.log');run([MS/'vlog.exe','-sv','tb.sv'],r,'compile.log')
log,seconds=run([MS/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1}; run -all; quit -code 0'],r,'simulation.log')
assert 'PASS FRAME_TRAFFIC_ORACLE' in log and not re.search(r'\*\* (Error|Fatal)',log)
result=dict(status='FRAME_TRAFFIC_ORACLE_PASS',readbeats=reads,writebeats=writes,responses=responses,descriptor_result_sha256=sha(d/'result.json'),tb_sha256=sha(r/'tb.sv'),canonical_sha256=sha(HERE/'variants/row_pair_arena/tb_fourway_full.sv'),log_sha256=sha(r/'simulation.log'),seconds=seconds)
(r/'result.json').write_text(json.dumps(result,indent=2));print(result)
