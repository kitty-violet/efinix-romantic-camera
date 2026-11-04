"""Survey physically nearby LUT3/LUT2 pairs for one native LOGIC cell.

Both outputs retain every true input dependency: unlike unrestricted packing,
the secondary output's two inputs must both be used by the primary output.
This avoids the historically troublesome logically-insensitive timing arcs.
No state, arithmetic interpretation, clock, pin or current source is edited.
"""
import argparse, collections, itertools, json, time
from pathlib import Path
from run_units import ROOT, sha
from video_service_graph import load
from camera_joint_native import OUT, used_inputs
from fixed_control_ddr_reuse_locations import locations
from survey_video_service_lut_collapse import value, ZERO


def main():
    p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True)
    p.add_argument('--physical',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();native=a.native.resolve();physical=a.physical.resolve();out=a.output.resolve()
    assert out.is_relative_to(ROOT/'tmp') and not out.exists();start=time.monotonic()
    proof=json.loads((native/'native-graph-audit.json').read_text());assert proof['status']=='SERVICE_NATIVE_RETAINED_GRAPH_AND_CUT_PASS'
    for f,h in proof['inputs'].items():assert sha(f)==h,f
    d=load();graft=json.loads((native/'graft.json').read_text());removed=set(graft['removed']);edits={}
    for r in graft['changed_loads']:edits.setdefault((r['cell'],r['pin']),{})[tuple(r['from_token'])]=tuple(r['to_token'])
    g={n:(k,{p:[edits.get((n,p),{}).get(t,t) for t in ts] for p,ts in pins.items()}) for n,(k,pins) in d['g'].items() if n not in removed}
    xy=locations(physical/'outflow/ti60f225_oob.out.placeloc');sdc=(physical/'ti60f225_oob.sdc').read_text()
    meta=json.loads((physical/'physical_inputs.json').read_text());board=Path(meta['base'])
    assert json.loads((board/'inputs.json').read_text())['native']==str(native)
    loads=collections.defaultdict(list)
    for n,(k,pins) in g.items():
        for pin,ts in pins.items():
            if pin not in OUT[k]:
                for t in ts:loads[t].append((n,pin))
    eligible={};two=collections.defaultdict(list)
    for n,(k,pins) in g.items():
        if k!='EFX_LUT4' or n not in xy or n in sdc or n.startswith('protected_inst/'):continue
        q=pins['O'][0]
        # Do not merge clocks, enables, resets or high-fanout physical buffers.
        if not loads[q] or len(loads[q])>16 or any(p in ('CLK','WCLK','RCLK','CLKA','CLKB','CE','SR') for _,p in loads[q]):continue
        active=sorted({t for ts in used_inputs(k,pins,d['ps'][n]).values() for t in ts if t[0]=='net'},key=str)
        if len(active) not in (2,3):continue
        ins=[pins.get('I'+str(i),[ZERO])[0] for i in range(4)]
        if any(t[0]=='literal' and t[1] not in ("1'b0","1'b1") for t in ins):continue
        # Do not use alias/constant support simplification to make a pair fit.
        if len({t for t in ins if t[0]=='net'})!=len(active):continue
        eligible[n]=dict(inputs=active,output=q,original_inputs=ins,mask=d['ps'][n]['LUTMASK'][1])
        if len(active)==2:two[tuple(active)].append(n)
    proposals=[]
    for n,o in eligible.items():
        for sub in itertools.combinations(o['inputs'],2):
            for peer in two.get(tuple(sub),[]):
                if peer==n:continue
                delta=sum(abs(xy[n][i]-xy[peer][i]) for i in (0,1))
                if delta<=12:proposals.append((delta,n,peer))
    used=set();rows=[]
    for distance,n,peer in sorted(proposals):
        if n in used or peer in used:continue
        o,pair=eligible[n],eligible[peer]
        inputs=pair['inputs']+[t for t in o['inputs'] if t not in pair['inputs']]
        assert len(inputs)<=3 and set(pair['inputs'])<=set(o['inputs'])
        primary=[];secondary=[]
        for v in range(8):
            values={t:v>>i&1 for i,t in enumerate(inputs)}
            primary.append(value(o['mask'],o['original_inputs'],values))
        for v in range(4):
            values={t:v>>i&1 for i,t in enumerate(pair['inputs'])}
            secondary.append(value(pair['mask'],pair['original_inputs'],values))
        mask=sum(b<<i for i,b in enumerate(primary))+sum(secondary[i%4]<<(8+i) for i in range(8))
        rows.append(dict(primary=n,secondary=peer,inputs=inputs,primary_output=o['output'],secondary_output=pair['output'],
                         primary_inputs=o['original_inputs'],secondary_inputs=pair['original_inputs'],
                         primary_mask=o['mask'],secondary_mask=pair['mask'],lutmask=mask,
                         distance=distance,positions=[xy[n],xy[peer]],fanouts=[len(loads[o['output']]),len(loads[pair['output']])]))
        used.update((n,peer))
    result=dict(status='NEARBY_DEPENDENCY_PRESERVING_LOGIC_PAIRS_SURVEY',native=str(native),native_sha256=sha(native/'native_source.v'),
                physical=str(physical),map_sha256=meta['base_map_sha256'],placement_sha256=sha(physical/'outflow/ti60f225_oob.out.placeloc'),
                sdc_sha256=sha(physical/'ti60f225_oob.sdc'),checker_sha256=sha(__file__),eligible=len(eligible),pairs=len(rows),rows=rows,
                primary_uses_every_secondary_input=True,hardware_access=False,source_edited=False,seconds=time.monotonic()-start)
    out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
