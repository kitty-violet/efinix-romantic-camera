"""Independent reversible-text and primitive proof of local RQ fanout."""
import argparse,itertools,json,pickle,re,time
from pathlib import Path
from run_units import ROOT,sha
from native_local_drivers import CELL,graph
from probe_full_native_split_map import params
from camera_joint_native import OUT
from audit_video_service_rq_fast import lut


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not (root/'native-graph-audit.json').exists()
    start=time.monotonic();receipt=root/'rq-local-edit.json';r=json.loads(receipt.read_text());base=Path(r['base'])
    previous=base/'native-graph-audit.json';parent=json.loads(previous.read_text())
    assert sha(previous)==r['parent_proof_sha256'] and parent['status']=='SERVICE_NATIVE_RETAINED_GRAPH_AND_CUT_PASS'
    assert sha(base/'native_source.v')==r['base_sha256'] and sha(r['placement'])==r['placement_sha256']
    for f,h in parent['inputs'].items():assert sha(f)==h,f
    key,g,ps=pickle.loads((base/'rq-audit-parse.pkl').read_bytes());assert key[0]==r['base_sha256']
    target=r['target'];kind,old=g[target];assert kind=='EFX_LUT4';token=old['O'][0]
    wanted={(n,p) for n,(k,pins) in g.items() for p,ts in pins.items() if p not in OUT[k] and token in ts}
    actual=set();rewrite={};comparisons=0
    for c in r['clones']:
        name=c['name'];assert name not in g
        cg=graph(c['cell']);cp=params(c['cell']);assert set(cg)==set(cp)=={name}
        assert cg[name][0]==kind and cp[name]==ps[target]
        pins=cg[name][1];assert pins['O']==[tuple(c['token'])]
        assert {p:ts for p,ts in pins.items() if p!='O'}=={p:ts for p,ts in old.items() if p!='O'}
        zero=('literal',"1'b0",None)
        ins=[old.get('I'+str(i),[zero])[0] for i in range(4)]
        cin=[pins.get('I'+str(i),[zero])[0] for i in range(4)]
        leaves=list(dict.fromkeys(t for t in ins if t[0]=='net'))
        for values in itertools.product((0,1,'x','z'),repeat=len(leaves)):
            v=dict(zip(leaves,values));assert lut(ps[target]['LUTMASK'][1],ins,v)==lut(cp[name]['LUTMASK'][1],cin,v);comparisons+=1
        for n,pin in c['loads']:
            assert (n,pin) in wanted and (n,pin) not in actual
            assert g[n][0]=='EFX_LUT4' and pin in ('I0','I1','I2','I3')
            actual.add((n,pin));rewrite.setdefault(n,{})[pin]=tuple(c['token'])
    assert actual==wanted and len(actual)==240
    records={x['name']:x for x in r['records']};assert set(records)==set(rewrite)
    for n,row in records.items():
        before=graph(row['before']);after=graph(row['after'])
        assert before[n]==g[n] and before[n][0]==after[n][0]=='EFX_LUT4'
        expected={p:([rewrite[n][p]] if p in rewrite[n] else ts) for p,ts in g[n][1].items()}
        assert after[n][1]==expected
    text=(root/'native_source.v').read_text();code=r['inserted_code'];decl=r['inserted_declarations']
    assert code=='\n'+''.join(c['cell'] for c in r['clones'])
    assert decl==''.join(c['declaration'] for c in r['clones'])
    assert all(c['declaration']=='    (* syn_keep="true" *) wire '+c['token'][1]+';\n' for c in r['clones'])
    assert text.count(code)==text.count(decl)==1
    restored=text.replace(code,'',1).replace(decl,'',1);seen=set()
    def restore(m):
        name=m[2].lstrip('\\')
        if name not in records:return m[0]
        assert name not in seen and m[0]==records[name]['after'];seen.add(name)
        return records[name]['before']
    restored=CELL.sub(restore,restored)
    assert seen==set(records) and restored==(base/'native_source.v').read_text()
    for f in base.glob('*.sv'):assert sha(root/f.name)==sha(f)
    inputs={**parent['inputs'],**{str(f):sha(f) for f in (Path(__file__).resolve(),receipt,root/'graft.json',root/'native_source.v',previous,Path(r['placement']),Path(__file__).with_name('prepare_video_service_rq_local.py'))}}
    result={**parent,'retained_primitives':parent['retained_primitives']+len(r['clones']),
            'localization_parent':str(base),'local_clones':len(r['clones']),'local_loads':len(actual),
            'local_four_state_comparisons':comparisons,'local_reversible_text_proof':True,
            'latency_added':0,'ordinary_placement_anchors_added':0,'inputs':inputs,
            'seconds':time.monotonic()-start,'hardware_access':False,'board_release':False}
    (root/'native-graph-audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ('status','retained_primitives','local_clones','local_loads','local_four_state_comparisons','latency_added','seconds')}))

if __name__=='__main__':main()
