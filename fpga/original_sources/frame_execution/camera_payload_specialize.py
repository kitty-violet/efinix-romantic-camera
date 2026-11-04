"""Independent native candidate, using the proved RAM-disabled don't-care only."""
import argparse, json, re
from pathlib import Path
from run_units import ROOT,sha
from camera_payload_owner import BASE,inspect_source
from native_local_drivers import CELL,raw_cells,graph,render_token,partition
from probe_full_native_split_map import params
from complete_native_logic_split import protected_blocks

def audit(before,after,groups):
    a,b=graph(before),graph(after);ap,bp=params(before),params(after)
    edits={(x['cell'],x['port']):('net',r['new_net'],None) for r in groups for x in r['loads']}
    new={r['new_cell']:r for r in groups}
    assert set(b)-set(a)==set(new) and not set(a)-set(b)
    for n,(k,pins) in a.items():
        assert b[n][0]==k and pins.keys()==b[n][1].keys()
        assert ap.get(n,{})==bp.get(n,{})
        for p,ts in pins.items():assert b[n][1][p]==([edits[(n,p)]] if (n,p) in edits else ts),(n,p)
    for n,r in new.items():
        parent=r['owner_ff']; assert b[n][0]==a[parent][0]=='EFX_FF'
        expected=dict(a[parent][1]);expected['Q']=[('net',r['new_net'],None)]
        assert b[n][1]==expected and bp[n]==ap[parent]
        assert expected['CLK']==[('net','process_clk_150',None)] and bp[n]['SR_SYNC']==(1,1)
    assert protected_blocks(before)==protected_blocks(after)
    for pattern in (r'^\s*(?:input|output|inout)\b.*$',r'^\s*assign\b.*$'):
        assert re.findall(pattern,before,re.M)==re.findall(pattern,after,re.M)
    return dict(status='PAYLOAD_SPECIALIZATION_GRAPH_PASS',original_cells=len(a),clones=len(new),
        changed_payload_muxes=len(edits),changed_other_pins=0,added_kinds={'EFX_FF':len(new)},
        latency_cycles_added=0,protected_blocks_byte_equal=True,hardware_access=False,physical_pass=False)

def transform(text,groups):
    raw=raw_cells(text);ps=params(text);edits={};lines=[]
    for i,r in enumerate(groups):
        n=f'CAMERA_PAYLOAD_OWNER_{i:03d}';wire=f'camera_payload_owner_{i:03d}'
        assert n not in raw
        r.update(new_cell=n,new_net=wire)
        for x in r['loads']:
            assert (x['cell'],x['port']) not in edits
            edits[(x['cell'],x['port'])]=wire
        pins=dict(raw[r['owner_ff']][1]);pins['Q']=wire
        lines.append(f'    (* syn_keep="true" *) wire {wire};')
        lines.append(f'    EFX_FF {n} ('+', '.join(f'.{p}({v})' for p,v in pins.items())+') /* local payload owner */ ;')
        for p,(w,v) in ps[r['owner_ff']].items():lines.append(f"    defparam {n}.{p} = {w}'h{v:x};")
    def change(m):
        n=m[2].lstrip('\\')
        def port(q):
            key=(n,q[1]);return f'.{q[1]}({edits[key]})' if key in edits else q[0]
        body=re.sub(r'\.(\w+)\(([^()]*)\)',port,m[3])
        return m[0][:m.start(3)-m.start()]+body+m[0][m.end(3)-m.start():]
    changed=CELL.sub(change,text)
    assert len(re.findall(r'^endmodule\b',changed,re.M))==1
    return re.sub(r'^endmodule\b',lambda _: '\n'.join(lines)+'\nendmodule',changed,flags=re.M)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args()
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
    source=BASE/'outflow/ti60f225_oob.map.v';text=source.read_text()
    _,_,banks,groups=inspect_source(text)
    placement=ROOT/'tmp/frame-execution/camera-checked-place2/outflow/ti60f225_oob.out.placeloc'
    positions={}
    for line in placement.read_text().splitlines():
        parts=line.split()
        if len(parts)==4 and not line.startswith('#'):positions[parts[0]]=tuple(map(int,parts[1:3]))
    local=[]
    for r in groups:
        index={(x['cell'],x['port']):x for x in r['loads']}
        assert all(n in positions for n,p in index)
        for sub in partition(list(index),positions):
            local.append({**r,'loads':[index[x] for x in sub]})
    groups=local;assert max(len(r['loads']) for r in groups)<=24
    changed=transform(text,groups)
    proof=audit(text,changed,groups)
    (root/'native_source.v').write_text(changed,encoding='utf8')
    (root/'groups.json').write_text(json.dumps(groups,indent=2),encoding='utf8')
    (root/'ram_enable_proof.json').write_text(json.dumps(banks,indent=2),encoding='utf8')
    first=groups[0]
    for label,bad in [('mux',changed.replace('.I2('+first['new_net']+')',".I2(1'b0)",1)),
                      ('clone',changed.replace('.Q('+first['new_net']+')',".Q(1'b0)",1))]:
        assert bad!=changed,label
        try:audit(text,bad,groups)
        except (AssertionError,KeyError):pass
        else:raise AssertionError('Accepted corruption '+label)
    proof.update(negative_checks=['mux','clone'],max_local_payload_fanout=max(len(r['loads']) for r in groups),
        inputs={str(f):sha(f) for f in (source,placement,Path(__file__),Path(__import__('camera_payload_owner').__file__))},
        files={p.name:sha(p) for p in root.iterdir() if p.is_file()},board_release=False)
    (root/'result.json').write_text(json.dumps(proof,indent=2),encoding='utf8');print(json.dumps(proof),flush=True)

if __name__=='__main__':main()
