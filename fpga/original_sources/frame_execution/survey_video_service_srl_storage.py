"""Read-only inventory of actual SRL histories, taps, widths and duplicates."""
import argparse, collections, json, re, time
from pathlib import Path
from run_units import ROOT, sha
from video_service_graph import load


def main():
    p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    native=a.native.resolve();out=a.output.resolve();assert out.is_relative_to(ROOT/'tmp') and not out.exists();start=time.monotonic()
    d=load();graft=json.loads((native/'graft.json').read_text());removed=set(graft['removed']);edits={}
    for e in graft['changed_loads']:edits.setdefault((e['cell'],e['pin']),{})[tuple(e['from_token'])]=tuple(e['to_token'])
    g={n:(k,{p:[edits.get((n,p),{}).get(t,t) for t in ts] for p,ts in pins.items()}) for n,(k,pins) in d['g'].items() if n not in removed}
    controls=collections.defaultdict(list);histories=collections.defaultdict(list);identical=collections.defaultdict(list);families=collections.Counter();rows=[]
    for n,(kind,pins) in g.items():
        if kind!='EFX_SRL8':continue
        par=d['ps'][n];control=(tuple(pins.get('CLK',[])),tuple(pins.get('CE',[])),tuple(pins.get('A',[])),tuple(sorted(par.items())))
        history=(tuple(pins.get('CLK',[])),tuple(pins.get('CE',[])),tuple(pins['D']),tuple(sorted(par.items())))
        full=history+(tuple(pins.get('A',[])),)
        controls[control].append(n);histories[history].append(n);identical[full].append(n)
        family=re.sub(r'\[\d+\]','[*]',n);families[family]+=1
        rows.append(dict(name=n,pins=pins,parameters=par))
    result=dict(status='READ_ONLY_SRL_STORAGE_INVENTORY',native=str(native),source_sha256=sha(native/'native_source.v'),
        cells=len(rows),families=families.most_common(),control_groups=[dict(width=len(v),control=k,cells=v) for k,v in sorted(controls.items(),key=lambda kv:-len(kv[1]))],
        repeated_histories=[v for v in histories.values() if len(v)>1],same_history_and_tap=[v for v in identical.values() if len(v)>1],
        repeated_exact_cells=sum(len(v)-1 for v in identical.values()),rows=rows,seconds=time.monotonic()-start,hardware_access=False,source_changed=False)
    out.write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ('status','cells','families','repeated_exact_cells','seconds')},indent=2))


if __name__=='__main__':main()
