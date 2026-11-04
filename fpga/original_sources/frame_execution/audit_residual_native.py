"""Close native warnings by exact sites and LUT truth tables; preserve raw logs."""
import argparse,collections,json,re
from pathlib import Path
from run_units import sha

def audit(root):
 r=json.loads((root/'result.json').read_text());m=json.loads((root/'inputs.json').read_text())
 assert sha(root/'inputs.json')==r['inputs_sha256']
 for x in m['inputs']:assert sha(x['snapshot'])==x['sha256']
 body=(root/'compile.log').read_text();counts=collections.Counter();pins=collections.Counter()
 net=next((Path(x['snapshot']) for x in m['inputs'] if x['snapshot'].endswith('.map.v')),None)
 text=net.read_text() if net else '';lines=text.splitlines()
 masks={n:int(v,16) for n,v in re.findall(r"defparam (\S+)\.LUTMASK = 16'h([0-9a-fA-F]+);",text)}
 allowed={'efx_dsp24.v':{95:1,109:1,189:1,190:1,193:1,215:1,280:2,327:1,328:1,331:1,332:1},
  'efx_ram10.v':{199:2,211:1,392:1},'efx_dpram10.v':{255:1,259:1,411:1,505:1},
  'shared_rq_oc8_cluster.sv':{61:1},'finish_rr_arbiter.sv':{41:1},'shared_pool_banked_fifo.sv':{52:1},
  'tb_residual_row_mapped.sv':{112:2}}
 sites=collections.Counter()
 for line in body.splitlines():
  if not line.startswith('%Warning'):continue
  v=re.fullmatch(r'%Warning-([A-Z0-9_]+): (.*):(\d+):(\d+): (.*)',line);assert v,line
  code,path,ln,col,msg=v.groups();ln=int(ln);name=Path(path.replace('\\','/')).name;counts[code]+=1
  if code=='PINMISSING':
   assert name=='residual_row_shared.map.v'
   pin=re.fullmatch(r"Instance has missing pin: '(\w+)'",msg)[1]
   kind,inst=re.search(r'(EFX_\w+)\s+(\S+)\s*\(',lines[ln-1]).groups()
   if kind=='EFX_LUT4':
    assert re.fullmatch('I[0-3]',pin);bit=int(pin[1]);mask=masks[inst]
    assert all((mask>>i&1)==(mask>>(i^(1<<bit))&1) for i in range(16))
   else:assert kind=='EFX_ADD' and pin in ('O','CO')
   pins[kind+'.'+pin]+=1
  else:
   assert code==('WIDTHTRUNC' if name=='tb_residual_row_mapped.sv' else 'WIDTHEXPAND')
   assert name in allowed and ln in allowed[name],line;sites[(name,ln)]+=1
 assert sites==collections.Counter({(f,n):v for f,d in allowed.items() for n,v in d.items()})
 assert dict(counts)==r['warning_codes']
 build=(root/'build.log').read_text();assert not re.search(r'\berror:',build,re.I)
 warnings=[x for x in build.splitlines() if 'warning:' in x]
 assert len(warnings)==2
 assert all(re.fullmatch(r".*include/verilated\.cpp:(78|79):10: warning: '(STDOUT|STDERR)_FILENO' redefined",x) for x in warnings)
 return dict(codes=dict(counts),proved_unused_pins=dict(pins),unknown=0,
  build_review='Two Windows runtime STDOUT/STDERR macro redefinitions; no DUT compilation warnings.',
  width_review=['Official DSP signed/unsigned extension is intentional.',
   'Official DSP/RAM enum strings zero-extend for comparisons; primitive sources unchanged.',
   'Unsigned owner/group indices compare against bounded nonnegative loop indices:0..7/0..3.',
   'TB scale_main is boolean0/1 in frozen vector cases; logical conversion preserves its meaning.'])

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--mapped',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
 result=dict(status='LOCAL_NATIVE_RESIDUAL_WARNING_AUDIT_PASS',source=audit(a.source),mapped=audit(a.mapped),
  evidence={str(x/f):sha(x/f) for x in (a.source,a.mapped) for f in ('result.json','inputs.json','compile.log','build.log')})
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
