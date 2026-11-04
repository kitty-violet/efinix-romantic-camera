"""Compile actual open Verilog with Questa; opaque IP is hash-bound, not decrypted."""
import argparse,json,os,re,time
from pathlib import Path
from run_units import ROOT,sha
from map_video_service_safe import run
p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
N=a.native.resolve();O=a.root.resolve();assert O.is_relative_to(ROOT/'tmp') and not O.exists();O.mkdir();start=time.monotonic()
text=(N/'native_source.v').read_text();blocks=re.findall(r'`pragma protect begin_protected.*?`pragma protect end_protected',text,re.S);assert blocks
open_text=re.sub(r'`pragma protect begin_protected.*?`pragma protect end_protected','// Opaque IP omitted only for open-syntax compile; official synthesis checks it.',text,flags=re.S)
# Efinity's exported netlist permits declarations after their first use.
# Questa's SV front end rejects that ordering. Hoist only plain uninitialized
# wire declarations in the syntax copy; no expression or production byte changes.
first=re.search(r'^\s*EFX_\w+\s+\S+\s*\(',open_text,re.M);assert first
prefix,tail=open_text[:first.start()],open_text[first.start():]
wire=re.compile(r'^[ \t]*(?:\(\*[^\r\n]*\*\)[ \t]*)?wire\b[ \t]*[^\r\n;=]+;[ \t]*$',re.M)
late=wire.findall(tail);tail=wire.sub('',tail)
legacy='reg [1:0] service_legacy_epoch=0;'
assert tail.count(legacy)==1;tail=tail.replace(legacy,'');late.append(legacy)
open_text=prefix+'\n'.join(late)+'\n'+tail
(O/'open_source.v').write_text(open_text,encoding='utf8')
os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat';MS=Path('E:/quatra/questa_fse/win64')
run([MS/'vlib.exe','work'],O,'vlib.log',30)
body,seconds=run([MS/'vlog.exe','-sv','open_source.v',*sorted(N.glob('*.sv'))],O,'compile.log',90)
assert 'Errors: 0' in body
inputs={str(f):sha(f) for f in [Path(__file__),N/'native_source.v',*sorted(N.glob('*.sv')),O/'open_source.v',O/'compile.log']}
result=dict(status='ACTUAL_OPEN_VERILOG_SYNTAX_PASS_NOT_IP_ELABORATION',inputs=inputs,opaque_blocks=len(blocks),syntax_only_hoisted_plain_wire_declarations=len(late),seconds=time.monotonic()-start,hardware_access=False,board_release=False)
(O/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
