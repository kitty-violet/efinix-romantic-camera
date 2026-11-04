"""Audit actual immutable frame descriptors before specializing any physical routing."""
import collections,json
from pathlib import Path
from run_units import ROOT,sha
root=ROOT/'tmp/frame-execution/joint-reorder-vectors1';manifest=json.loads((root/'frame_manifest.json').read_text())
assert sha(root/'program.hex')==manifest['files']['program.hex']
rows=collections.defaultdict(collections.Counter);outs=collections.defaultdict(collections.Counter)
counts=collections.Counter();maps=collections.Counter();cmds=0
for line in (root/'program.hex').read_text().splitlines():
 x=int(line,16);f=[(x>>(32*i))&0xffffffff for i in range(32)]
 if f[0]!=2:continue
 cmds+=1;flags=f[6];pw=bool(flags&1);stride=bool(flags&8);rgb=pw and f[28]==7
 name='rgb' if rgb else 'stride' if stride else 'pw_fused' if flags&16 else 'pw' if pw else 'nn_dw' if flags&2 else 'stem' if flags&4 else 'dw'
 n=f[3] if pw else f[3]*2+1 if stride else f[3]+2
 planes=tuple((f[10]>>(2*i))&3 for i in range(n));bases=f[11]|(f[12]<<32);seen=[[] for _ in range(3)];mapping=[]
 for i,p in enumerate(planes):
  b=(bases>>(10*i))&1023
  if b not in seen[p]:seen[p].append(b)
  assert len(seen[p])<=2
  mapping.append(p*2+seen[p].index(b))
 rows[name][planes]+=1;maps[tuple(mapping)]+=1;counts[name]+=1
 outs[name][tuple((f[13]>>(2*i))&3 for i in range(f[3]))]+=1
result=dict(status='DESCRIPTOR_LAYOUT_ONLY_NOT_RTL_EQUIVALENCE',commands=cmds,program_sha256=sha(root/'program.hex'),
 modes={k:dict(commands=counts[k],input_plane_patterns=[dict(pattern=list(p),count=c) for p,c in v.most_common()],
 output_plane_patterns=[dict(pattern=list(p),count=c) for p,c in outs[k].most_common()]) for k,v in rows.items()},
 rowmap_patterns=[dict(pattern=list(p),count=c) for p,c in maps.most_common()],
 notes=['Static observation of all actual descriptors; generic RTL is unchanged.',
 'Any narrowed routing requires a hardware rejection guard and compiler invariant proof, including clipped/nearest duplicate rows.'])
out=ROOT/'output/toolchain/frame_descriptor_layout_2026-10-01.json';assert not out.exists()
out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(commands=cmds,rowmap_patterns=len(maps),modes={k:dict(commands=counts[k],input_patterns=len(v),output_patterns=len(outs[k])) for k,v in rows.items()})))
