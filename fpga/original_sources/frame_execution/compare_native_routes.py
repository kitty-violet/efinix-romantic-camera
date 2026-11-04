"""Compare actual same-map, same-placement routed STA endpoint surveys."""
import argparse,json,collections
from pathlib import Path
from run_units import sha

def read(root):
    file=root/'data_arc_analysis.json';data=json.loads(file.read_text())
    assert sha(data['report'])==data['report_sha256']
    assert sha(root/'result.json')==data['source_result_sha256']
    inp=json.loads((root/'inputs.json').read_text());base=Path(inp['base'])
    return data,base

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True)
    p.add_argument('--new',type=Path,required=True);a=p.parse_args()
    old,ob=read(a.old);new,nb=read(a.new)
    for name in ('outflow/ti60f225_oob.map.v','outflow/ti60f225_oob.place','outflow/ti60f225_oob.vdb','ti60f225_oob.sdc'):
        assert sha(ob/name)==sha(nb/name),name
    def by_endpoint(data):
        out={(r['clock'],r['end']):r for r in data['paths']}
        assert len(out)==len(data['paths'])
        return out
    before=by_endpoint(old);after=by_endpoint(new);common=set(before)&set(after)
    changes=[dict(clock=k[0],end=k[1],old_ns=before[k]['slack_ns'],new_ns=after[k]['slack_ns'],
        delta_ns=round(after[k]['slack_ns']-before[k]['slack_ns'],6),
        old_start=before[k]['start'],new_start=after[k]['start'],
        old_longest=before[k]['longest'],new_longest=after[k]['longest']) for k in sorted(common)]
    result=dict(status='SAME_MAP_PLACEMENT_CONSTRAINTS_ROUTE_COMPARISON_NOT_RELEASE',
        inputs={str(a.old/'data_arc_analysis.json'):sha(a.old/'data_arc_analysis.json'),
                str(a.new/'data_arc_analysis.json'):sha(a.new/'data_arc_analysis.json')},
        old_negative=old['sampled_negative'],new_negative=new['sampled_negative'],
        old_worst_ns=min(r['slack_ns'] for r in old['paths']),new_worst_ns=min(r['slack_ns'] for r in new['paths']),
        common_endpoints=len(common),new_sample_only=len(set(after)-set(before)),
        old_sample_only=len(set(before)-set(after)),
        improved=sum(x['delta_ns']>0 for x in changes),worse=sum(x['delta_ns']<0 for x in changes),
        crossed_to_pass=sum(x['old_ns']<0<=x['new_ns'] for x in changes),
        crossed_to_fail=sum(x['new_ns']<0<=x['old_ns'] for x in changes),
        worst_new=sorted(changes,key=lambda x:x['new_ns'])[:40],
        most_improved=sorted(changes,key=lambda x:x['delta_ns'],reverse=True)[:20],
        most_worsened=sorted(changes,key=lambda x:x['delta_ns'])[:20],
        changes=changes,limitation='Each report is capped at1000 worst endpoints; sample membership can change. No extrapolated PASS.',
        script_sha256=sha(__file__),hardware_access=False)
    dest=a.new/'route_comparison.json';assert not dest.exists()
    dest.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('changes','worst_new','most_improved','most_worsened')}))
