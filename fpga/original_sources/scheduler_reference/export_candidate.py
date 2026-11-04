"""Repack only SRAM bases/order/dependencies; verify all arithmetic fields unchanged."""
from pathlib import Path
import ast,hashlib,json,random,shutil,zlib
from analyze_trace import load_functions
R=Path(__file__).resolve().parent
accesses=load_functions('compile_schedule.py',{'units','accesses'})['accesses']
ns=dict(P=Path('E:/fpga'),ast=ast,random=random)
tree=ast.parse((R/'compile_descriptors.py').read_text('utf8'))
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in('schemas','pack','replay')],type_ignores=[]),'compile_descriptors.py','exec'),ns)

def decode(raw,schema):
 word=int.from_bytes(raw,'little');offset=0;out={}
 for name,width in schema:out[name]=(word>>offset)&((1<<width)-1);offset+=width
 return out

def emit(commands,reference,oldbin):
 schemas=ns['schemas']();prev={};idx=[0,0]
 for c in reference:
  e=int(c['kind']!='compute');size=16 if e else 64
  prev[c['original_command']]=(c,decode(oldbin[e][idx[e]*size:(idx[e]+1)*size],schemas[e]));idx[e]+=1
 writes=[None]*12288;reads=[[0]*12288 for _ in range(2)];counts=[0,0];out=[bytearray(),bytearray()];streams=[[],[]];regions=[]
 unchanged_count=0
 for c in commands:
  rr,ww,pad=accesses(c);rs={u for u,_ in rr};ws={u for u,_ in ww}|pad;regions.append((rs,ws))
  e=int(c['kind']!='compute');peer=1-e;dep=0
  for u in rs|ws:
   if writes[u]and writes[u][0]==peer:dep=max(dep,writes[u][1])
   if u in ws:dep=max(dep,reads[peer][u])
  counts[e]+=1
  for u in rs:reads[e][u]=counts[e]
  for u in ws:writes[u]=(e,counts[e])
  streams[e].append(dict(peer_completed=dep,program_index=c['command']))
  old,oldvalues=prev[c['original_command']];values=dict(oldvalues);values.update(peer_completed=dep,f31=c['command'])
  if e:
   assert {k:v for k,v in c.items()if k not in('command','row')}=={k:v for k,v in old.items()if k not in('command','row')}
   assert {k:v for k,v in c['row'].items()if k!='base'}=={k:v for k,v in old['row'].items()if k!='base'}
   values['f5']=c['row']['base'];allowed={'peer_completed','f31','f5'}
  else:
   assert {k:v for k,v in c.items()if k not in('command','inputs','outputs','skip')}=={k:v for k,v in old.items()if k not in('command','inputs','outputs','skip')}
   for rows,lo,hi in [('inputs',11,12),('outputs',14,15),('skip',21,22)]:
    assert len(c[rows])==len(old[rows])
    for a,b in zip(c[rows],old[rows]):assert {k:v for k,v in a.items()if k!='base'}=={k:v for k,v in b.items()if k!='base'}
    bits=sum(d['base']<<(10*y)for y,d in enumerate(c[rows]));values['f'+str(lo)]=bits&0xffffffff;values['f'+str(hi)]=bits>>32
   values['program_index']=c['command'];allowed={'peer_completed','program_index','f31','f11','f12','f14','f15','f21','f22'}
  assert all(values[k]==v for k,v in oldvalues.items()if k not in allowed)
  unchanged_count+=sum(k not in allowed for k in values)
  packed=ns['pack'](values,schemas[e],3 if e else 0x55,16 if e else 64)
  assert decode(packed,schemas[e])==values
  out[e].extend(packed)
 trials=[ns['replay'](streams,regions,seed)for seed in(71,391,1027)]
 return out,dict(counts=counts,unchanged_fields_checked=unchanged_count,replays=trials)

def main():
 reference=json.loads((R/'schedule/program.json').read_text('utf8'))
 oldbin=[(R/'vectors'/n).read_bytes()for n in('compute.bin','dma.bin')]
 check,_=emit(reference,reference,oldbin);assert check==oldbin,'Identity round-trip failed'
 commands=json.loads((R/'allocation-candidate/program.json').read_text('utf8'))
 out,proof=emit(commands,reference,oldbin)
 dest=R/'candidate-vectors';dest.mkdir(exist_ok=False)
 for p in(R/'vectors').glob('*.bin'):
  if p.name not in('compute.bin','dma.bin'):shutil.copy2(p,dest/p.name)
 for name,payload in zip(('compute.bin','dma.bin'),out):(dest/name).write_bytes(payload)
 (dest/'crc.txt').write_text(' '.join(f'{zlib.crc32(b):08x}'for b in out)+'\n','ascii')
 files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())for p in dest.iterdir()}
 proof.update(status='DESCRIPTOR_STATIC_AND_ABSTRACT_REPLAY_PASS_NOT_RTL_PASS',files=files,identity_roundtrip=True,hardware_access=False,
  unchanged='All weights, quantization/arithmetic, DDR addresses, compute order, full3x3 taps, RGB publication order, command counts and golden vectors unchanged. Only local SRAM base allocation, DMA order and required dependencies vary.')
 (dest/'result.json').write_text(json.dumps(proof,indent=2),'utf8');print(json.dumps(proof,indent=2))

if __name__=='__main__':main()
