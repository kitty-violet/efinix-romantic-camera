"""Actual 20-KiB serialization and isolated six-bank layout for clean skip."""
import argparse,copy,importlib.util,json,sys
from pathlib import Path
from . import compile_component_row_pair as original
from . import compile_b1_capacity as diagnostic_source
from . import capacity_fixed_headers as headers
from .clean_skip_capacity_net import plan,budget
from .runtime import file_hash,write_json
from .train_luma_regions import ROOT,RUN


def fused_plan():
    result=[]
    for row in plan()['nodes']:
        n=copy.deepcopy(row)
        if n['kind']=='residual_add':
            p=result.pop();assert p['id']==n['inputs'][0]
            n.update(kind='pw_fused',parameter_node=p['id'],inputs=[p['inputs'][0],n['inputs'][1]],input_channels=p['input_channels'],input_shape_hwc=p['input_shape_hwc'],terms=p['terms'],macs=p['macs'])
        result.append(n)
    result[-1]['id']='rgb_head';return dict(nodes=result)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='clean-skip-allocation-v1');a=ap.parse_args();root=ROOT/'runs'/RUN/a.name
    diagnostic_source.plan=plan;headers.plan=plan;headers.budget=budget;diagnostic_source.coefficient_image=headers.coefficient_image;diagnostic_source.diagnostic(root)
    text=Path(original.__file__).read_text();needle="    assert chosen,('ROW_PAIR_CAPACITY',ti,trials)";assert text.count(needle)==1
    text=text.replace(needle,"    if chosen is None:\n        from .capacity_bank_search import search\n        chosen,report=search(records,local,placement);trials.append(dict(node_bank_search=report))\n"+needle)
    src=root/'frozen_compiler.py';src.write_text(text);spec=importlib.util.spec_from_file_location('network_ab.training._clean_skip_probe',src);compiler=importlib.util.module_from_spec(spec);spec.loader.exec_module(compiler);sys.path.insert(0,str(original.HW))
    trials=[]
    for width in [64,48,32]:
        row=dict(width=width,positions=[])
        try:
            for i,x in enumerate([0,width,1280-width]):
                _,r=compiler.compile_one(fused_plan(),i,x,width);row['positions'].append(r);print('CLEAN_SKIP_ALLOCATION',width,x,r['high'],flush=True)
            row['status']='THREE_POSITIONS_PASS_NOT_FULL_FRAME'
        except (AssertionError,MemoryError) as e:row.update(status='FAIL',error=str(e));print('CLEAN_SKIP_ALLOCATION_FAIL',width,str(e)[:200],flush=True)
        trials.append(row);write_json(root/'allocation.json',dict(budget=budget(),trials=trials,compiler_sha256=file_hash(src),graph_sha256=file_hash(Path(__file__).with_name('clean_skip_capacity_net.py')),synthetic_parameters=True,rtl=False,fit=False,timing=False))
        if row['status'].startswith('THREE'):break


if __name__=='__main__':main()
