"""Classify every Questa serial-cone warning against the exact native primitive."""
import argparse,collections,json,re
from pathlib import Path
from run_units import EF,sha
from probe_full_native_split_map import params

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
 result=json.loads((root/'result.json').read_text());assert result['status']=='VIDEO_SERVICE_INDEXED_SERIAL_NATIVE_PASS'
 out=root/'warning-port-audit.json';assert not out.exists()
 bodies={n:(root/n).read_text() for n in ('old.sv','new.sv')};lines={n:s.splitlines() for n,s in bodies.items()};ps={n:params(s) for n,s in bodies.items()}
 warnings=re.findall(r'^# \*\* Warning: (.*)$',(root/'simulate.log').read_text(),re.M)
 short=set();missing=set();rows=[];counts=collections.Counter()
 for message in warnings:
  m=re.search(r'\\(old|new)\.sv\((\d+)\): \((vopt-\d+)\)',message);assert m,message
  file=m[1]+'.sv';line=int(m[2]);key=(file,line);code=m[3]
  assert code in ('vopt-2685','vopt-2718'),message
  if code=='vopt-2685':short.add(key);continue
  pin=re.search(r"Missing connection for port '([^']+)'",message)[1];missing.add(key)
  c=re.match(r'\s*(EFX_\w+)\s+(\S+)\s*\(',lines[file][line-1]);assert c,(file,line)
  kind,name=c.groups();name=name.lstrip('\\')
  if kind=='EFX_ADD':assert pin in ('O','CO');reason='unused output'
  elif kind=='EFX_SRL8':assert pin=='Q7';reason='unused cascade output'
  else:
   assert kind=='EFX_LUT4' and re.fullmatch(r'I[0-3]',pin),(kind,pin)
   bit=int(pin[1]);mask=ps[file][name]['LUTMASK'][1]
   assert all((mask>>i&1)==(mask>>(i^(1<<bit))&1) for i in range(16)),(name,pin,hex(mask))
   reason='Boolean function independent of omitted input'
  rows.append(dict(file=file,line=line,cell=name,port=pin,reason=reason));counts[kind+'.'+pin]+=1
 assert short==missing and len(rows)+len(short)==len(warnings)
 files=[root/n for n in ('old.sv','new.sv','simulate.log','result.json')]+[Path(__file__)]
 for n in ('efx_lut4.v','efx_add.v','efx_srl8.v','efx_ff.v'):
  vendor=EF/'sim_models/verilog'/n;assert sha(root/n)==sha(vendor);files.extend([root/n,vendor])
 record=dict(status='ALL_SERIAL_NATIVE_WARNINGS_CLASSIFIED_PASS',warnings=len(warnings),actual_missing_ports=len(rows),counts=dict(counts),rows=rows,evidence={str(f):sha(f) for f in files},hardware_access=False,board_release=False)
 out.write_text(json.dumps(record,indent=2));print(json.dumps({k:record[k] for k in ('status','warnings','actual_missing_ports','counts')}))
if __name__=='__main__':main()
