"""Read existing C2 evidence only; do not infer causality from one worst path."""
import argparse,json,re
from pathlib import Path
from run_units import ROOT,sha
from audit_global_place import metrics

def paths(physical):
    source=physical/'outflow/ti60f225_oob.timing.rpt'
    if not source.exists():return []
    text=source.read_text().split('Path Detail Report (process_clk_150 vs process_clk_150)',1)[1]
    text=text.split('Path Detail Report (',1)[0]
    rows=[]
    for block in re.split(r'\+\+\+\+ Path \d+ \++',text)[1:]:
        def value(name):return re.search(r'^'+re.escape(name)+r'\s*:\s*(.*?)\s*$',block,re.M)[1]
        data=block.split('\nData Path\n',1)[1].split('\nCapture Clock Path\n',1)[0]
        delays=[(model,float(delay)) for model,delay in re.findall(r'^.*?\s+(EFX_\w+|net)\s+([\d.]+)\s+[\d.]+\s+\d+\s+\(',data,re.M)]
        assert delays
        net=sum(d for model,d in delays if model=='net')
        total=sum(d for _,d in delays)
        rows.append(dict(start=value('Path Begin'),end=value('Path End'),
            slack_ns=float(value('Slack').split()[0]),logic_levels=int(value('Logic Level')),
            net_delay_ns=round(net,3),data_delay_ns=round(total,3),
            net_fraction=round(net/total,4)))
    assert len(rows)==10,len(rows)
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();assert out.is_relative_to(ROOT/'output/toolchain') and not out.exists()
    root=ROOT/'tmp/frame-execution';cases={}
    for name,folder in [('original','local-decode-place1'),('cached','cached-head-place1'),('feedback','feedback-select-place1')]:
        physical=root/folder;m=metrics(physical)
        cases[name]=dict(physical=str(physical),metrics=m,reported_process_setup_top10=paths(physical))
    old,new=cases['original'],cases['feedback']
    before=root/'connected-board-local-decode1';after=root/'connected-board-feedback-select1'
    untouched=['fixed_window_dot8x32.sv','fixed_dma_program.sv','fixed_production_mac.sv']
    # Locate the actual DMA owner source rather than assume its filename.
    untouched=[n for n in untouched if (before/'assembly'/n).exists()]
    for file in (after/'assembly').glob('*.sv'):
        if 'module cnn_tensor_dma' in file.read_text() and file.name not in untouched:untouched.append(file.name)
    same={}
    for name in untouched:
        left=before/'assembly'/name;right=after/'assembly'/name
        assert sha(left)==sha(right),name
        same[name]=dict(original=str(left),current=str(right),sha256=sha(left))
    for case in cases.values():
        file=Path(case['physical'])/'outflow/ti60f225_oob.timing.rpt'
        if file.exists():case['routed_report']=dict(path=str(file),sha256=sha(file))
    selected=root/'feedback-select-route1/selected.json';terminal=json.loads(selected.read_text())
    result=dict(status='C2_GLOBAL_COST_REGRESSION_CONFIRMED_CAUSAL_ABLATION_PENDING',
        cases=cases,current_wns_ns=terminal['wns_ns'],current_whs_ns=terminal['whs_ns'],
        current_sta_evidence=dict(path=str(selected),sha256=sha(selected)),unchanged_arithmetic_sources=same,
        xlr_delta=new['metrics']['resources']['XLRs']['used']-old['metrics']['resources']['XLRs']['used'],
        wirelength_change_pct=100*(new['metrics']['weighted_hpwl_per_net']/old['metrics']['weighted_hpwl_per_net']-1),
        conclusions=[
          'Whole-design resource and wirelength costs increased; final WNS regressed from -0.660 to -0.706 ns.',
          'Original stored-to-feedback paths are absent from the new reported top ten; this does not prove every such path passes.',
          'Worst current path has one LUT level; long routed nets dominate. Adding more generic pipeline storage is not justified by this evidence.',
          'MAC RTL is byte-identical while its physical context changed. Functional equivalence cannot establish timing equivalence.',
          'Two structural edits and a repack/replacement changed together. Their separate causal effects remain unmeasured.',
          'Only ten process-clock paths are present in the existing full-tool report. Their sum is not full-design TNS or total violation count.',
          'Retain original C2 and both failed candidates; compare a minimal feedback-selector-only variant before adding more cache registers.'
        ],hardware_access=False,new_implementation_started=False,board_release=False,
        generator_sha256=sha(__file__))
    out.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:result[k] for k in ('status','xlr_delta','wirelength_change_pct','current_wns_ns','current_whs_ns')}))
    print(json.dumps(cases['feedback']['reported_process_setup_top10'][:3]))
if __name__=='__main__':main()
