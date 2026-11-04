"""Factor the measured loader ready chain without changing cycles or controls."""
import argparse,itertools,json,re,sys,traceback
from pathlib import Path
from run_units import ROOT,sha
from native_local_drivers import graph,raw_cells,CELL,render_token
from probe_full_native_split_map import params
from complete_native_logic_split import protected_blocks
BASE=ROOT/'tmp/frame-execution/connected-board-camera-payload1'
PREFIX='video_processing_pipeline/pipeline/cnn/core/dut/'
ZERO=('literal',"1'b0",None)
def net(n,i=None):return ('net',n,i)
def mask(fn):return sum(int(fn(*[(v>>i)&1 for i in range(4)]))<<v for v in range(16))
def circuit(text):
 g=graph(text);ps=params(text);drivers={p['O'][0]:n for n,(k,p) in g.items() if k=='EFX_LUT4'}
 return g,ps,drivers
def cone(g,drivers,target,boundary):
 seen=set();order=[]
 def visit(t):
  if t in boundary or t[0]=='literal':return
  assert t in drivers,('UNEXPECTED_BOUNDARY',t)
  n=drivers[t]
  if n in seen:return
  seen.add(n)
  for p,ts in g[n][1].items():
   if p!='O':
    for s in ts:visit(s)
  order.append(n)
 visit(target);return order
def eval_cone(g,ps,cells,fixed,target):
 values=dict(fixed);values[ZERO]=0
 for n in cells:
  pins=g[n][1];v=sum(values[pins.get('I'+str(i),[ZERO])[0]]<<i for i in range(4))
  values[pins['O'][0]]=(ps[n]['LUTMASK'][1]>>v)&1
 return values[target]
def plan(text):
 g,ps,drivers=circuit(text)
 # Original request = clk & ((held & keep) | d & (!e | c & (wsr & a & b))).
 boundary=[net(PREFIX+'engine/wsr',5),net('n63369'),net('n63370_2'),net('n63372_2'),net('n63375_2'),net('n63387'),net(PREFIX+'loader/held'),net('n63255'),net('u_ddr3_top/controller/u_efx_ddr3_soft_controller/genblk1.inst_ddr3_axi/inst_efx_ddr3/top_mc/clk_rstn')]
 request=net('n63389');x=net('camera_ready_product');y=net('camera_ready_existing')
 definitions={
  'CAMERA_READY_PRODUCT':([boundary[i] for i in (1,2,3,5)],x,mask(lambda a,b,c,d:a and b and c and d)),
  'CAMERA_READY_EXISTING':([boundary[i] for i in (5,4,6,7)],y,mask(lambda d,e,h,k:(d and not e) or (h and k))),
  drivers[request]:([boundary[8],y,x,boundary[0]],request,mask(lambda clk,y,x,w:clk and (y or (x and w))))}
 cases=[dict(name='loader_request',boundary=boundary,target=request)]
 # Keep the original reset-dependent mac-ready term as an independent input.
 # This balances associative products without Boolean absorption that would
 # hide X values in the original four-state circuit.
 w=net(PREFIX+'engine/wsr',5)
 wb=[net('~'+PREFIX+'engine/router/n6'),net('n63338_2'),net(PREFIX+'abort_event'),net('n63335_2'),net('n63366_2'),net('n63288_2'),net('video_processing_pipeline/pipeline/cnn/core/header/active_groups',2),net('n63290'),net('n63275_2'),net('n63276_2')]
 wg=net('camera_ready_group');wh=net('camera_ready_conditions');wq=net('camera_ready_queue')
 definitions.update({
  'CAMERA_READY_GROUP':([wb[5],wb[6],wb[7],ZERO],wg,mask(lambda a,g,c,z:c and (g or not a))),
  'CAMERA_READY_CONDITIONS':([wb[2],wb[3],wb[4],wg],wh,mask(lambda abort,cache,start,group:not abort and not cache and start and group)),
  'CAMERA_READY_QUEUE':([wb[8],wb[9],ZERO,ZERO],wq,mask(lambda a,b,c,d:not a and b)),
  drivers[w]:([wb[0],wb[1],wh,wq],w,mask(lambda reset,mac,condition,q:not reset and (not (mac and condition) or q)))})
 cases.append(dict(name='shared_weight_ready',boundary=wb,target=w))
 for b in range(5):
  boundary2=[request,net('n63396_2')]+[net(PREFIX+'loader/issue_bank',i) for i in range(3)]
  target=net('video_processing_pipeline/pipeline/cnn/core/n8304',0) if b==0 else net(PREFIX+'loader/take',b)
  pre=net(f'camera_ready_bank_{b}')
  definitions[f'CAMERA_READY_BANK_{b}']=([boundary2[1],*boundary2[2:]],pre,mask(lambda ready,b0,b1,b2,b=b:ready and (b0+2*b1+4*b2)==b))
  definitions[drivers[target]]=([request,pre,ZERO,ZERO],target,mask(lambda r,p,a,b:r and p))
  cases.append(dict(name=f'bank{b}_take',boundary=boundary2,target=target))
 for c in cases:
  c['old_cells']=cone(g,drivers,c['target'],set(c['boundary']))
 return definitions,cases
def transform(text,definitions):
 raw=raw_cells(text);blocks=[]
 def ports(ins,out):return ', '.join(f'.I{i}({render_token(t)})' for i,t in enumerate(ins))+f', .O({render_token(out)})'
 def change(m):
  n=m[2].lstrip('\\')
  if n not in definitions:return m[0]
  ins,out,value=definitions[n];assert m[1]=='EFX_LUT4'
  return f"    EFX_LUT4 {n} ({ports(ins,out)}) /* exact ready factoring */ ;"
 changed=CELL.sub(change,text)
 for n,(ins,out,value) in definitions.items():
  if n in raw:
   pattern=r'(defparam\s+'+re.escape(n)+r'\.LUTMASK\s*=\s*)16\'h[0-9a-fA-F]+'
   changed,count=re.subn(pattern,lambda m:m[1]+f"16'h{value:04x}",changed);assert count==1,n
  else:
   blocks.extend([f'    (* syn_keep="true" *) wire {out[1]};',f'    EFX_LUT4 {n} ({ports(ins,out)}) /* exact ready factoring */ ;',f"    defparam {n}.LUTMASK = 16'h{value:04x};"])
 assert len(re.findall(r'^endmodule\b',changed,re.M))==1
 return re.sub(r'^endmodule\b',lambda m:'\n'.join(blocks)+'\nendmodule',changed,flags=re.M)
def audit(before,after,definitions,cases):
 a,ap,ad=circuit(before);b,bp,bd=circuit(after)
 assert set(a)<=set(b)
 assert set(b)-set(a)==set(definitions)-set(a)
 for n in a:
  if n not in definitions:assert a[n]==b[n] and ap.get(n,{})==bp.get(n,{}),n
 for n,(ins,out,value) in definitions.items():
  assert b[n]==('EFX_LUT4',{**{f'I{i}':[t] for i,t in enumerate(ins)},'O':[out]})
  assert bp[n]=={'LUTMASK':(16,value)}
 assert protected_blocks(before)==protected_blocks(after)
 for pattern in (r'^\s*(?:input|output|inout)\b.*$',r'^\s*assign\b.*$'):
  assert re.findall(pattern,before,re.M)==re.findall(pattern,after,re.M)
 total=0
 for c in cases:
  c['new_cells']=cone(b,bd,c['target'],set(c['boundary']))
  for bits in itertools.product((0,1),repeat=len(c['boundary'])):
   fixed=dict(zip(c['boundary'],bits))
   assert eval_cone(a,ap,c['old_cells'],fixed,c['target'])==eval_cone(b,bp,c['new_cells'],fixed,c['target']),(c['name'],bits)
   total+=1
 return dict(status='READY_FACTOR_GRAPH_AND_EXHAUSTIVE_BINARY_PASS',checks=total,added_luts=len(set(b)-set(a)),changed_existing_luts=len(set(definitions)&set(a)),added_cycles=0,added_ff=0,added_ram=0,added_dsp=0,hardware_access=False,board_release=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();out=a.root.resolve();assert out.is_relative_to(ROOT/'tmp') and not out.exists();out.mkdir()
 source=BASE/'outflow/ti60f225_oob.map.v';text=source.read_text();definitions,cases=plan(text)
 changed=transform(text,definitions);r=audit(text,changed,definitions,cases)
 # Corruption must be rejected at the exact graph gate.
 needle="defparam CAMERA_READY_PRODUCT.LUTMASK = 16'h8000;"
 assert changed.count(needle)==1
 bad=changed.replace(needle,"defparam CAMERA_READY_PRODUCT.LUTMASK = 16'h0000;")
 try:audit(text,bad,definitions,cases)
 except AssertionError:pass
 else:raise AssertionError('corruption not rejected')
 (out/'native_source.v').write_text(changed,encoding='utf8');(out/'cases.json').write_text(json.dumps(cases,indent=2))
 (out/'definitions.json').write_text(json.dumps(definitions,indent=2))
 r.update(inputs={str(source):sha(source),str(Path(__file__)):sha(__file__)},files={f.name:sha(f) for f in out.iterdir() if f.is_file()},negative_checks=1,four_state_pass=False)
 (out/'result.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
if __name__=='__main__':
 try:main()
 except Exception as e:
  out=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
  if out.is_dir():(out/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc(),script_sha256=sha(__file__),hardware_access=False),indent=2))
  raise
