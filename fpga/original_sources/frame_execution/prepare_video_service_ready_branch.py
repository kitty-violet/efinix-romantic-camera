"""Balance only the critical AND branch; retain the other shared reductions."""
import argparse,itertools,json,re,shutil
from pathlib import Path
from run_units import ROOT,sha
from native_local_drivers import CELL,graph,render_token
from probe_full_native_split_map import params
from audit_video_service_rq_fast import lut

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 base=a.base.resolve();out=a.root.resolve();assert out.is_relative_to(ROOT/'tmp') and not out.exists()
 pp=base/'native-graph-audit.json';parent=json.loads(pp.read_text())
 assert parent['retained_primitives']==54577 and parent['local_clones']==16
 for f,h in parent['inputs'].items():assert sha(f)==h,f
 text=(base/'native_source.v').read_text();names=['LUT__132942','LUT__132950']
 blocks={m[2].lstrip('\\'):m[0] for m in CELL.finditer(text) if m[2].lstrip('\\') in names}
 lines=[re.search(r'^\s*defparam '+n+r'\.LUTMASK\s*=[^;]*;',text,re.M)[0] for n in names]
 g=graph('\n'.join(blocks.values()));ps=params('\n'.join(lines))
 assert all(g[n][0]=='EFX_LUT4' and ps[n]['LUTMASK']==(16,0x8000) for n in names)
 ap,bp=[g[n][1] for n in names];assert bp['I0']==ap['O']
 newnet=('net','service_ready_preconditions',None);name='SERVICE_READY_PRECONDITIONS'
 assert name not in text and newnet[1] not in text
 newa={**ap,'I2':bp['I1'],'I3':bp['I2'],'O':[newnet]}
 newb={**bp,'I0':[newnet],'I1':ap['I2'],'I2':ap['I3']}
 leaves=list(dict.fromkeys(ap['I0']+ap['I1']+ap['I2']+ap['I3']+bp['I1']+bp['I2']+bp['I3']));assert len(leaves)==7
 def evaluate(a,b,v):
  v=dict(v);v[a['O'][0]]=lut(0x8000,[a['I'+str(i)][0] for i in range(4)],v)
  return lut(0x8000,[b['I'+str(i)][0] for i in range(4)],v)
 count=0
 for bits in itertools.product((0,1,'x','z'),repeat=7):
  v=dict(zip(leaves,bits));assert evaluate(ap,bp,v)==evaluate(newa,newb,v);count+=1
 def cell(n,p):return '\n    EFX_LUT4 '+n+' ('+', '.join('.'+k+'('+render_token(ts[0])+')' for k,ts in p.items())+') /* balanced critical branch */ ;\n'
 insert=cell(name,newa)+f"    defparam {name}.LUTMASK = 16'h8000;\n"
 declaration='    (* syn_keep="true" *) wire service_ready_preconditions;\n'
 old=blocks[names[1]];new=cell(names[1],newb);assert text.count(old)==1
 changed=text.replace(old,new,1);pos=CELL.search(changed).start();changed=changed[:pos]+declaration+changed[pos:]
 pos=changed.index('    protected_cell protected_inst (');changed=changed[:pos]+insert+changed[pos:]
 assert changed.replace(declaration,'',1).replace(insert,'',1).replace(new,old,1)==text
 out.mkdir()
 for f in base.glob('*.sv'):shutil.copy2(f,out/f.name);assert sha(f)==sha(out/f.name)
 (out/'native_source.v').write_text(changed,encoding='utf8')
 rp=out/'ready-branch-edit.json';rp.write_text(json.dumps(dict(base=str(base),base_sha256=sha(base/'native_source.v'),before=old,after=new,insert=insert,declaration=declaration,checks=count,added_luts=1,added_state=0,latency_added=0,ordinary_fixed_locations_added=0,late_fault_lut_levels_before=2,late_fault_lut_levels_after=1,early_other_conditions_extra_level=1),indent=2))
 graft=json.loads((base/'graft.json').read_text());graft['ready_branch']=dict(base=str(base),manifest=str(rp),manifest_sha256=sha(rp));graft['inputs'][str(Path(__file__).resolve())]=sha(__file__)
 (out/'graft.json').write_text(json.dumps(graft,indent=2))
 inputs={**parent['inputs'],**{str(f):sha(f) for f in (pp,Path(__file__).resolve(),rp,out/'graft.json',out/'native_source.v')}}
 proof={**parent,'retained_primitives':54578,'ready_branch_parent':str(base),'ready_branch_checks':count,'ready_branch_reversible_text':True,'inputs':inputs,'hardware_access':False,'board_release':False}
 (out/'native-graph-audit.json').write_text(json.dumps(proof,indent=2))
 print(json.dumps(dict(status=proof['status'],checks=count,retained=54578,source_sha256=sha(out/'native_source.v'))))
if __name__=='__main__':main()
