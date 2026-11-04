"""Independent full-source/Boolean-cut proof, including a failing CEC control."""
import argparse,copy,json,pickle,re,subprocess,sys,time
from pathlib import Path
from run_units import ROOT,sha
from native_local_drivers import graph
from probe_full_native_split_map import params
from complete_native_logic_split import protected_blocks
from continuous_control_native import connections
from control_remap_cut import boundaries
V=ROOT/'tmp/board-test/video-service-review1';sys.path.insert(0,str(V))
from blif_control import read,write
p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);a=p.parse_args();N=a.native.resolve()
r=json.loads((N/'control-remap-edit.json').read_text());P=Path(r['parent']);start=time.monotonic()
assert r['source_sha256']==sha(N/'native_source.v') and r['parent_sha256']==sha(P/'native_source.v')
for f,h in r['inputs'].items():assert sha(f)==h
key,old,op=pickle.loads((P/'zero-source-parse.pkl').read_bytes());assert key==sha(P/'native_source.v')
text=(N/'native_source.v').read_text();actual=graph(text);ap=params(text)
owned=set(r['source_cells']);added=set(r['new_cells'])
assert all(old[n][0]=='EFX_LUT4' for n in owned) and all(actual[n][0]=='EFX_LUT4' for n in added)
assert set(old)-owned==set(actual)-added
assert all(old[n]==actual[n] and op[n]==ap[n] for n in set(old)-owned)
assert protected_blocks(text)==protected_blocks((P/'native_source.v').read_text())
for pat in (r'^\s*(?:input|output|inout)\b.*$',r'^\s*assign\b.*$'):
 assert [x.strip() for x in re.findall(pat,text,re.M)]==[x.strip() for x in re.findall(pat,(P/'native_source.v').read_text(),re.M)]
for f in P.glob('*.sv'):assert sha(f)==sha(N/f.name)
tg,boundary,ba=boundaries(old,(P/'native_source.v').read_text(),owned)
tg2,boundary2,ba2=boundaries(actual,text,added)
expected_outputs={old[n][1]['O'][0] for n in tg}
new_outputs={actual[n][1]['O'][0] for n in tg2}
assert expected_outputs==new_outputs,dict(missing=sorted(expected_outputs-new_outputs,key=str)[:8],extra=sorted(new_outputs-expected_outputs,key=str)[:8])
assert boundary2<=boundary and boundary==set(map(tuple,r['cut_inputs']))
bt={n:tuple(t) for n,t in r['blif_tokens'].items()};rev={t:n for n,t in bt.items()};ref=read(N/'imported.blif')
assert len(bt)==len(rev) and set(rev[t] for t in expected_outputs)<=set(ref['outputs'])
fresh=dict(inputs=ref['inputs'],outputs=ref['outputs'],nodes={});truth=0
for n in sorted(added):
 pins=actual[n][1];o=rev[pins['O'][0]];ts=[pins.get('I'+str(i),[('literal',"1'b0",None)])[0] for i in range(4)]
 ins=list(dict.fromkeys(t for t in ts if t[0]=='net'));mask=0
 for b in range(1<<len(ins)):
  env={t:b>>i&1 for i,t in enumerate(ins)}
  idx=sum((env[t] if t[0]=='net' else int(t[1][-1]))<<i for i,t in enumerate(ts))
  mask|=(ap[n]['LUTMASK'][1]>>idx&1)<<b;truth+=1
 fresh['nodes'][o]=dict(inputs=[rev[t] for t in ins],mask=mask)
assert fresh==ref,'Emitted EFX graph must re-export to the independently read exact BLIF'
write(N/'actual-efx.blif',fresh)
(N/'actual-equivalence.abc').write_text('cec original.blif actual-efx.blif\n')
abc=Path(__file__).with_name('run_installed_abc.py')
logs=[]
for name,expected in [('actual-equivalence','Networks are equivalent.'),('negative','Networks are NOT EQUIVALENT.')]:
 log=N/(name+'.log');assert not log.exists()
 with log.open('w') as f:
  cp=subprocess.run([sys.executable,str(abc),'--script',str(N/(name+'.abc'))],stdout=f,stderr=subprocess.STDOUT,timeout=60)
 assert cp.returncode==0
 body=log.read_text();assert expected in body,(name,body)
 if name=='actual-equivalence':assert 'NOT EQUIVALENT' not in body
 logs.append(log)
paths=[Path(__file__),Path(__file__).with_name('control_remap_cut.py'),abc,V/'blif_control.py',P/'native_source.v',P/'native-graph-audit.json',P/'weight-zero-independent-audit.json',P/'endpoint-batch-independent-audit-v2.json',N/'native_source.v',N/'control-remap-edit.json',N/'original.blif',N/'actual-efx.blif',N/'actual-equivalence.abc',N/'tampered.blif',N/'negative.abc',*logs]
evidence={str(f):sha(f) for f in paths}
proof=dict(status='CONTROL_REMAP_FULL_SOURCE_CEC_PASS_NOT_PHYSICAL',source_sha256=sha(N/'native_source.v'),parent_sha256=key,
 primitive_count=len(actual),old_luts=len(owned),new_luts=len(added),boundary_inputs=len(boundary),boundary_outputs=len(ref['outputs']),actual_external_outputs=len(expected_outputs),actual_lut_truth_rows=truth,
 all_outside_cut_and_sequential_state_exact=True,all_external_consumers_retained=True,latency_added=0,cec_all_boolean_inputs=True,tampered_output_rejected=True,
 arbitrary_unknown_controls_proven=False,scope='Compositional exact step-equivalence to58: all state/arithmetic/ports/protected IP unchanged; every cut output equivalent for all boundary Boolean inputs. Reset/interface simulations remain separate.',
 inputs=evidence,hardware_access=False,board_release=False,seconds=time.monotonic()-start)
dest=N/'control-remap-independent-audit.json';dest.write_text(json.dumps(proof,indent=2))
parentwp=json.loads((P/'weight-zero-independent-audit.json').read_text());cache=json.loads((ROOT/'tmp/frame-execution/video-service-parallel-group-source1/cuts.json').read_text())['new_cells']
changed_cache=[n for n in cache if actual.get(n)!=old[n] or ap.get(n)!=op[n]]
parentwp.update(source_sha256=sha(N/'native_source.v'),weight_cache_unchanged_from_parent=not changed_cache,
 scope='Parent58 actual A tests preserved; all changed combinational control functions proved exactly at their complete boundary. No weight data or sequential state changed.',
 cache_changed_cells=changed_cache,control_remap_proof=str(dest),inputs={**parentwp['inputs'],**evidence,str(dest):sha(dest)})
(N/'weight-zero-independent-audit.json').write_text(json.dumps(parentwp,indent=2))
na=json.loads((P/'native-graph-audit.json').read_text());na.update(retained_primitives=len(actual),control_remap_scope=proof['status'],inputs={**na['inputs'],**evidence,str(dest):sha(dest),str(N/'graft.json'):sha(N/'graft.json')})
(N/'native-graph-audit.json').write_text(json.dumps(na,indent=2));(N/'zero-source-parse.pkl').write_bytes(pickle.dumps((sha(N/'native_source.v'),actual,ap)))
print(json.dumps({k:v for k,v in proof.items() if k!='inputs'}));print('CACHE_CELLS_CHANGED',len(changed_cache))
