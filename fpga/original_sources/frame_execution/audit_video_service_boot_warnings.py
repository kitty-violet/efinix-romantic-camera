"""Resolve every mapped-boot missing-port warning against the exact LUT truth."""
import argparse,collections,json,re
from pathlib import Path
from run_units import EF,sha
from probe_full_native_split_map import params

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
 out=root/'warning-port-audit.json';assert not out.exists()
 mapped=root/'mapped_boot.sv';text=mapped.read_text();lines=text.splitlines();ps=params(text)
 body=(root/'simulate.log').read_text();warnings=re.findall(r'^# \*\* Warning: \(([^)]+)\) (.*)$',body,re.M)
 counts=collections.Counter();rows=[];short_lines=set();missing_lines=set()
 assert len(warnings)==968
 for code,message in warnings:
  assert code in ('vsim-3017','vsim-3722'),(code,message)
  if code=='vsim-3017':
   short_lines.add(int(re.search(r'mapped_boot.sv\((\d+)\)',message)[1]));continue
  m=re.search(r'mapped_boot.sv\((\d+)\).*Missing connection for port \'([^\']+)\'',message);assert m,message
  line=int(m[1]);pin=m[2];missing_lines.add(line)
  cell=re.match(r'\s*(EFX_\w+)\s+(\S+)\s*\(',lines[line-1]);assert cell,(line,lines[line-1])
  kind,name=cell.groups();name=name.lstrip('\\')
  if kind=='EFX_ADD':assert pin in ('CO','O');reason='unused adder output'
  else:
   assert kind=='EFX_LUT4' and re.fullmatch(r'I[0-3]',pin),(kind,pin)
   bit=int(pin[1]);mask=ps[name]['LUTMASK'][1]
   assert all((mask>>i&1)==(mask>>(i^(1<<bit))&1) for i in range(16)),(name,pin,hex(mask))
   reason='LUT function independent of omitted input'
  counts[(kind,pin)]+=1;rows.append(dict(line=line,cell=name,kind=kind,port=pin,reason=reason))
 assert missing_lines==short_lines and len(rows)+len(short_lines)==len(warnings)
 evidence={str(f):sha(f) for f in (mapped,root/'simulate.log',root/'result.json',Path(__file__))}
 for name in ('efx_lut4.v','efx_add.v'):
  f=root/name;vendor=EF/'sim_models/verilog'/name;assert sha(f)==sha(vendor);evidence[str(f)]=sha(f);evidence[str(vendor)]=sha(vendor)
 result=dict(status='EVERY_MISSING_PORT_UNUSED_OUTPUT_OR_BOOLEAN_INDEPENDENT_INPUT_PASS',warnings=len(warnings),
  actual_missing_ports=len(rows),counts={f'{k[0]}.{k[1]}':v for k,v in counts.items()},rows=rows,evidence=evidence)
 out.write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ('status','warnings','actual_missing_ports','counts')}))
if __name__=='__main__':main()
