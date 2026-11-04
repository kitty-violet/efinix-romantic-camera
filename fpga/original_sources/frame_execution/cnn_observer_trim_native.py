"""Remove retired diagnostics only; preserve functional cells and live levels.

Independent candidate. This is neither an FPGA release nor a download entry.
The observer transport retains its framing/handshake. Retired fields must be
masked by the deployment-bound decoder, never interpreted as measurements.
"""
import argparse, collections, json, re
from pathlib import Path
from native_local_drivers import CELL, REF, graph, rewrite_expression
from camera_joint_native import OUT
from probe_full_native_split_map import params
from complete_native_logic_split import protected_blocks
from run_units import ROOT, HERE, sha

PARENT=ROOT/'tmp/frame-execution/connected-board-cnn-minimal1'
PREFIX='cpu/system_hardJtag_debug_logic_jtagBridge/observation/'
RETIRED=set(range(512,632))|set(range(639,832))

def transform(text):
    g=graph(text);ps=params(text)
    captures={}
    for group in ('held_payload','snapshot'):
        captures[group]={int(m[1]):n for n,(k,p) in g.items() if k=='EFX_FF' and
            (m:=re.fullmatch(re.escape(PREFIX+group)+r'\[(\d+)\]~FF',n)) and int(m[1]) in RETIRED}
    assert captures['held_payload'].keys()==captures['snapshot'].keys()
    assert len(captures['held_payload'])==272
    cut={(n,'D') for n in captures['held_payload'].values()}
    drivers=collections.defaultdict(set);loads=collections.defaultdict(set)
    for n,(k,pins) in g.items():
        for pin,ts in pins.items():
            for t in ts:
                if t[0]!='net':continue
                (drivers[t] if pin in OUT.get(k,()) else loads[t]).add(n if pin in OUT.get(k,()) else (n,pin))
    outside=CELL.sub('',text)
    outside=re.sub(r'^\s*defparam\b[^;]*;','',outside,flags=re.M)
    outside=re.sub(r'^\s*wire\b[^;]*;','',outside,flags=re.M)
    external={m['id'].lstrip('\\') for m in REF.finditer(outside) if m['id']}
    todo=[t for n,pin in cut for t in g[n][1][pin] if t[0]=='net'];seen=set();candidate=set()
    while todo:
        t=todo.pop()
        if t in seen:continue
        seen.add(t)
        if len(drivers[t])!=1:continue
        n=next(iter(drivers[t]));k,pins=g[n]
        if k not in {'EFX_FF','EFX_ADD','EFX_LUT4'} or n.startswith('protected_inst/'):continue
        candidate.add(n)
        todo.extend(t for p,ts in pins.items() if p not in OUT[k] for t in ts if t[0]=='net')
    keep_sync={n for i in range(6) for n in (f'clock_probe/meta[{i}]~FF',f'ddr_clock_diagnostic[{120+i}]~FF')}
    assert keep_sync<=g.keys()
    exclusive=set(candidate)-keep_sync
    while True:
        reject={n for n in exclusive for pin,ts in g[n][1].items() if pin in OUT[g[n][0]] for t in ts
            if t[0]=='net' and (t[1] in external or len(drivers[t])!=1 or
                any(edge not in cut and edge[0] not in exclusive for edge in loads[t]))}
        if not reject:break
        exclusive-=reject
    capture_names=set(captures['held_payload'].values())|set(captures['snapshot'].values())
    assert not capture_names & exclusive
    aliases={g[n][1]['Q'][0]:('literal',"1'b0",None) for n in capture_names}
    for bit,h in captures['held_payload'].items():
        s=captures['snapshot'][bit]
        assert g[s][1]['D']==g[h][1]['Q']
        assert loads[g[h][1]['Q'][0]]=={(s,'D')}
    removed=exclusive|capture_names
    # Independent proof: no exclusive source output may escape the retired
    # capture D ports, and no protected cell/port is changed by substitution.
    for n in exclusive:
        for p,ts in g[n][1].items():
            if p not in OUT[g[n][0]]:continue
            for t in ts:
                assert t[1] not in external
                assert all(edge in cut or edge[0] in exclusive for edge in loads[t])
    def replace(m):
        n=m[2].lstrip('\\')
        if n in removed:return ''
        body=re.sub(r'\.(\w+)\(([^()]*)\)',lambda p:'.'+p[1]+'('+rewrite_expression(p[2],aliases)+')',m[3])
        return m[0][:m.start(3)-m.start()]+body+m[0][m.end(3)-m.start():]
    result=CELL.sub(replace,text)
    result=re.sub(r'^\s*defparam\s+(\S+?)\s*\.\w+\s*=[^;]*;',
        lambda m:'' if m[1].lstrip('\\') in removed else m[0],result,flags=re.M)
    ng=graph(result);nps=params(result)
    assert set(ng)==set(g)-removed
    changed=[]
    for n,(k,pins) in ng.items():
        expect={p:[aliases.get(t,t) for t in ts] for p,ts in g[n][1].items()}
        assert k==g[n][0] and pins==expect and nps.get(n,{})==ps.get(n,{})
        if pins!=g[n][1]:
            assert k=='EFX_LUT4' and not n.startswith('protected_inst/')
            changed.append(n)
    assert protected_blocks(text)==protected_blocks(result)
    for pattern in (r'^\s*(?:input|output|inout)\b.*$',r'^\s*assign\b.*$'):
        assert re.findall(pattern,text,re.M)==re.findall(pattern,result,re.M)
    for i in range(6):
        for n in (f'clock_probe/meta[{i}]~FF',f'ddr_clock_diagnostic[{120+i}]~FF'):
            assert ng[n]==g[n] and nps[n]==ps[n]
    for n in ng:
        if n.startswith('protected_inst/'):
            assert ng[n]==g[n]
    # Synthesis merges equal bits across the payload. Check the real serial
    # destination of every changed input, not merely its FF's chosen name.
    serial_ends=set();todo=list(changed);walked=set()
    while todo:
        n=todo.pop()
        if n in walked:continue
        walked.add(n)
        for pin,ts in g[n][1].items():
            if pin not in OUT[g[n][0]]:continue
            for t in ts:
                assert t[1] not in external
                for sink,port in loads[t]:
                    if g[sink][0]=='EFX_FF':
                        m=re.fullmatch(re.escape(PREFIX)+r'shifter\[(\d+)\]~FF',sink)
                        assert port=='D' and m and int(m[1])-64 in RETIRED,(sink,port,t)
                        serial_ends.add(int(m[1])-64)
                    else:
                        assert g[sink][0]=='EFX_LUT4',(sink,port)
                        todo.append(sink)
    proof=dict(status='RETIRED_OBSERVER_CONE_AND_CAPTURE_REMOVAL_GRAPH_PASS',
        source_only_cells=len(exclusive),source_only_kinds=dict(collections.Counter(g[n][0] for n in exclusive)),
        capture_cells=len(capture_names),removed_cells=sorted(removed),
        removed_kinds=dict(collections.Counter(g[n][0] for n in removed)),
        serializer_luts_with_retired_inputs=changed,
        only_affected_serial_payload_bits=sorted(serial_ends),
        preserved_clock_sync_pairs=6,preserved_live_payload_bits=list(range(632,639)),
        retired_payload_ranges=[[512,632],[639,832]],transport_schema='OFP5',
        diagnostic_profile='cnn-observer-live-levels-only-v1',
        retired_fields=['relative_clock_counts_and_ticks','reset_age_and_edges','startup_trace'],
        protected_blocks_byte_equal=True,functional_ports_and_parameters_unchanged=True,
        latency_cycles_added=0,hardware_access=False,physical_pass=False,board_release=False)
    return result,proof

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    out=a.root.resolve();assert out.is_relative_to(ROOT/'tmp') and not out.exists()
    src=PARENT/'outflow/ti60f225_oob.map.v'
    assert sha(src)=='2796ceb91151326977b2dd86f3b1474d582020b0d1030d9bcde487ef71f8a1a2'
    result,proof=transform(src.read_text())
    out.mkdir();(out/'native_source.v').write_text(result,encoding='utf8')
    proof.update(parent=str(PARENT),parent_map_sha256=sha(src),source_sha256=sha(out/'native_source.v'),
        inputs={str(f):sha(f) for f in (src,Path(__file__),HERE/'native_local_drivers.py',HERE/'camera_joint_native.py',HERE/'probe_full_native_split_map.py')})
    (out/'result.json').write_text(json.dumps(proof,indent=2),encoding='utf8')
    print(json.dumps({k:proof[k] for k in ('status','source_only_cells','capture_cells','removed_kinds')}),flush=True)

if __name__=='__main__':main()
