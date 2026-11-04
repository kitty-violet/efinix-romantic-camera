"""Verify actual pair wiring/parameters, unchanged state, RTL and encrypted IP."""
import argparse, json, re, time
from pathlib import Path
from run_units import ROOT, sha
from audit_continuous_vectors import expand_graph
from probe_full_native_split_map import params
from complete_native_logic_split import protected_blocks
from video_service_graph import load
from native_local_drivers import CELL


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp')
    start=time.monotonic();graft=json.loads((root/'graft.json').read_text());c=graft['logic_pairs'];base=Path(c['base'])
    assert sha(base/'native_source.v')==c['base_sha256'] and sha(c['proof'])==c['proof_sha256'] and sha(c['survey'])==c['survey_sha256']
    pf=json.loads(Path(c['proof']).read_text());sv=json.loads(Path(c['survey']).read_text())
    assert c['pairs']==[dict(sv['rows'][i],pair_name=f'SERVICE_LOGIC_PAIR_{i:04d}') for i in pf['accepted']]
    assert pf['vectors_per_pair']==256 and pf['outputs_per_pair']==2 and pf['status']=='OFFICIAL_LOGIC_PAIR_FOUR_STATE_FILTER_COMPLETED'
    for f,h in pf['inputs'].items():assert sha(f)==h
    old_audit=json.loads((base/'native-graph-audit.json').read_text())
    for f,h in old_audit['inputs'].items():assert sha(f)==h
    d=load();oldgraft=json.loads((base/'graft.json').read_text());edits={}
    for r in oldgraft['changed_loads']:edits.setdefault((r['cell'],r['pin']),{})[tuple(r['from_token'])]=tuple(r['to_token'])
    removed=set(oldgraft['removed']);old={n:(k,{p:[edits.get((n,p),{}).get(t,t) for t in ts] for p,ts in pins.items()}) for n,(k,pins) in d['g'].items() if n not in removed}
    s=(root/'native_source.v').read_text();new=expand_graph(s);np=params(s);removed_pair={r[k] for r in c['pairs'] for k in ('primary','secondary')};pairs={r['pair_name']:r for r in c['pairs']}
    assert set(new)==set(old)-removed_pair|pairs.keys() and len(new)==graft['retained_primitive_count']
    zero=('literal',"1'b0",None)
    for n,(kind,pins) in new.items():
        if n in pairs:
            r=pairs[n];expected={f'I{i}':[tuple(r['inputs'][i]) if i<len(r['inputs']) else zero] for i in range(4)}
            expected.update(CI=[zero],O=[tuple(r['primary_output'])],P=[tuple(r['secondary_output'])],CO=[],FCO=[])
            assert kind=='EFX_COMB4' and pins==expected and np[n]=={'LUTMASK':(16,r['lutmask']),'MODE':'"LOGIC"'},(n,kind,pins,np[n])
            for role in ('primary','secondary'):
                name=r[role];assert old[name][0]=='EFX_LUT4' and d['ps'][name]['LUTMASK'][1]==r[role+'_mask']
                assert [old[name][1].get('I'+str(i),[zero])[0] for i in range(4)]==list(map(tuple,r[role+'_inputs']))
                assert old[name][1]['O']==[tuple(r[role+'_output'])]
        else:assert (kind,pins)==old[n] and np.get(n)==d['ps'].get(n),n
    assert protected_blocks(s)==protected_blocks((base/'native_source.v').read_text())
    for f in base.glob('*.sv'):assert sha(f)==sha(root/f.name)
    def exterior(t):
        t=CELL.sub('',t);t=re.sub(r'^\s*defparam\s+[^;]*;','',t,flags=re.M);return re.sub(r'\s+','',t)
    assert exterior(s)==exterior((base/'native_source.v').read_text())
    result=dict(status='SERVICE_NATIVE_RETAINED_GRAPH_AND_CUT_PASS',retained_primitives=len(new),removed_primitives=len(graft['removed']),
        explicit_changed_loads=len(graft['changed_loads']),equivalent_logic_pairs=len(pairs),four_state_output_comparisons=512*len(pairs),
        ordinary_retained_primitives_exact=len(new)-len(pairs),state_and_latency_unchanged=True,protected_blocks_unchanged=True,existing_assignments_unchanged=True,
        inputs={str(f):sha(f) for f in [Path(__file__),base/'native-graph-audit.json',base/'native_source.v',root/'native_source.v',root/'graft.json',Path(c['proof']),Path(c['survey'])]},
        seconds=time.monotonic()-start,hardware_access=False,board_release=False)
    out=root/'native-graph-audit.json';assert not out.exists();out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='inputs'},indent=2))


if __name__=='__main__':main()
