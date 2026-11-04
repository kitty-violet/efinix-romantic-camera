"""Exact generated-DW warning audit; no wildcard warning suppression."""
import collections,re
from pathlib import Path
from run_units import sha,EF
def audit(root):
 import json
 m=json.loads((root/'inputs.json').read_text())
 for r in m['inputs']:assert sha(r['snapshot'])==r['sha256']
 net=root/'source/dw_column_cluster.map.v';text=net.read_text();lines=text.splitlines();body=(root/'compile.log').read_text()
 masks={n:int(v,16) for n,v in re.findall(r"defparam (\S+)\.LUTMASK = 16'h([0-9a-fA-F]+);",text)}
 names=collections.Counter(re.findall(r'\bOpen_\d+\b',text));outputs={}
 for kind,inst,block in re.findall(r'^\s*(EFX_\w+)\s+(\S+)\s*\((.*?)\)\s*(?:/\*.*?\*/\s*)?;',text,re.M|re.S):
  for pin,arg in re.findall(r'\.(\w+)\s*\(([^)]*)\)',block):
   for name in re.findall(r'\bOpen_\d+\b',arg):
    assert name not in outputs;outputs[name]=(kind,pin)
 allowed={'efx_dsp24.v':{95:1,109:1,189:1,190:1,193:1,215:1,280:2,327:1,328:1,331:1,332:1},
 'efx_ram10.v':{199:2,211:1,392:1},'efx_dpram10.v':{255:1,259:1,411:1,505:1},
 'efx_dsp48.v':{710:2,819:1,820:1,823:1,824:1,333:1,339:1,340:1,343:1,344:1,345:1,370:1,103:1,110:1,124:1,131:1}}
 for name in allowed:assert sha(root/'source'/name)==sha(EF/'sim_models/verilog'/name)
 codes=collections.Counter();pins=collections.Counter();sites=collections.Counter();opens=collections.Counter()
 for line in body.splitlines():
  if not line.startswith('%Warning'):continue
  match=re.fullmatch(r'%Warning-([A-Z0-9_]+): (.*):(\d+):(\d+): (.*)',line);assert match,line
  code,path,ln,col,msg=match.groups();ln=int(ln);name=Path(path.replace('\\','/')).name;codes[code]+=1
  if code=='PINMISSING':
   assert name==net.name;pin=re.fullmatch(r"Instance has missing pin: '(\w+)'",msg)[1]
   kind,inst=re.search(r'(EFX_\w+)\s+(\S+)\s*\(',lines[ln-1]).groups()
   if kind=='EFX_LUT4':
    assert re.fullmatch(r'I[0-3]',pin);bit=int(pin[1]);mask=masks[inst]
    assert all((mask>>i&1)==(mask>>(i^(1<<bit))&1) for i in range(16))
   else:assert (kind,pin) in {('EFX_ADD','O'),('EFX_ADD','CO'),('EFX_DSP48','CASCOUT'),('EFX_DSP48','OVFL'),('EFX_DSP24','CASCOUT'),('EFX_SRL8','Q7')}
   pins[kind+'.'+pin]+=1
  elif code=='IMPLICIT':
   assert name==net.name;signal=re.fullmatch(r"Signal definition not found, creating implicitly: '(Open_\d+)'",msg)[1]
   assert names[signal]==1 and signal in lines[ln-1]
   assert outputs[signal] in {('EFX_DSP24','O'),('EFX_DSP48','O'),('EFX_RAM10','RDATA')},(signal,outputs.get(signal))
   opens['.'.join(outputs[signal])]+=1
  else:
   assert code=='WIDTHEXPAND' and name in allowed and ln in allowed[name],line;sites[(name,ln)]+=1
 assert sites==collections.Counter({(f,n):v for f,d in allowed.items() for n,v in d.items()})
 assert sum(opens.values())==len(names)==1184
 assert 'Verilator: Built from' in body and not re.search(r'^%Error',body,re.M)
 return dict(codes=dict(codes),proved_unused_pins=dict(pins),single_occurrence_unused_outputs=dict(opens),unknown=0,
  width_review='Exact official primitive enum/string and signed arithmetic width sites; official sources unchanged.')
