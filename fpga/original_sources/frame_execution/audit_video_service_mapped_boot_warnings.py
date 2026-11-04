"""Classify actual vopt/vsim boot warnings without dropping either format."""
import argparse,collections,json,re
from pathlib import Path
from run_units import EF,sha
from probe_full_native_split_map import params

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
 result=json.loads((root/'result.json').read_text());assert result['status']=='VIDEO_SERVICE_ACTUAL_MAP_BOOT_PASS'
 out=root/'warning-port-audit.json';assert not out.exists()
 mapped=root/'mapped_boot.sv';text=mapped.read_text();lines=text.splitlines();ps=params(text)
 body=(root/'simulate.log').read_text();warnings=re.findall(r'^# \*\* Warning: (.*)$',body,re.M)
 assert warnings and re.search(r'^# Errors: 0, Warnings: '+str(len(warnings))+r'\s*$',body,re.M)
 assert 'Errors: 0, Warnings: 0' in (root/'compile.log').read_text()
 short=set();missing=set();rows=[];counts=collections.Counter()
 for message in warnings:
  code=re.search(r'\((vopt|vsim)-(\d+)\)',message);assert code,message
  code=code[0][1:-1];assert code in ('vopt-2685','vopt-2718','vsim-3017','vsim-3722'),message
  lm=re.search(r'mapped_boot\.sv\((\d+)\)',message);assert lm,message
  line=int(lm[1]);assert 1<=line<=len(lines)
  if code in ('vopt-2685','vsim-3017'):short.add(line);continue
  pin=re.search(r"Missing connection for port '([^']+)'",message)[1];missing.add(line)
  c=re.match(r'\s*(EFX_\w+)\s+(\S+)\s*\(',lines[line-1]);assert c,(line,lines[line-1])
  kind,name=c.groups();name=name.lstrip('\\')
  if kind=='EFX_ADD':assert pin in ('O','CO');reason='unused adder output'
  else:
   assert kind=='EFX_LUT4' and re.fullmatch(r'I[0-3]',pin),(kind,pin)
   bit=int(pin[1]);mask=ps[name]['LUTMASK'][1]
   assert all((mask>>i&1)==(mask>>(i^(1<<bit))&1) for i in range(16)),(name,pin)
   reason='full LUT truth table independent of omitted weak-zero input'
  rows.append(dict(line=line,cell=name,kind=kind,port=pin,reason=reason));counts[kind+'.'+pin]+=1
 assert short==missing and len(rows)+len(short)==len(warnings)
 files=[mapped,root/'simulate.log',root/'compile.log',root/'result.json',Path(__file__)]
 for name in ('efx_lut4.v','efx_add.v','efx_ff.v','efx_ram10.v'):
  vendor=EF/'sim_models/verilog'/name;assert sha(root/name)==sha(vendor);files.extend([root/name,vendor])
 record=dict(status='EVERY_MISSING_PORT_UNUSED_OUTPUT_OR_BOOLEAN_INDEPENDENT_INPUT_PASS',warnings=len(warnings),actual_missing_ports=len(rows),counts=dict(counts),rows=rows,evidence={str(f):sha(f) for f in files},hardware_access=False,board_release=False)
 out.write_text(json.dumps(record,indent=2));print(json.dumps({k:record[k] for k in ('status','warnings','actual_missing_ports','counts')}))
if __name__=='__main__':main()
