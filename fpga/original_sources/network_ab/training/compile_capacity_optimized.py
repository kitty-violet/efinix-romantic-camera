"""Isolated compiler derivative that additionally searches per-node bank offsets."""
import argparse,importlib.util,json,sys
from pathlib import Path
from . import compile_component_row_pair as original
from .compile_b1_capacity import fused_plan
from .capacity_fixed_headers import coefficient_image
from .b1_capacity_component_net import budget,plan
from .runtime import file_hash,write_json
from .train_luma_regions import ROOT,RUN


def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--name',required=True);p.add_argument('--width',type=int,default=64);p.add_argument('--probe',action='store_true');a=p.parse_args()
    base=ROOT/'runs'/RUN;source=Path(original.__file__);text=source.read_text();needle="    assert chosen,('ROW_PAIR_CAPACITY',ti,trials)";assert text.count(needle)==1
    text=text.replace(needle,"    if chosen is None:\n        from .capacity_bank_search import search\n        chosen,report=search(records,local,placement);trials.append(dict(node_bank_search=report))\n"+needle)
    srcdir=base/(a.name+'-compiler');srcdir.mkdir(exist_ok=False);derived=srcdir/'derived_capacity_compiler.py';derived.write_text(text)
    name='network_ab.training._derived_capacity_compiler';spec=importlib.util.spec_from_file_location(name,derived);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.fused_plan=fused_plan;module.coefficient_image=coefficient_image;sys.path.insert(0,str(original.HW))
    if a.probe:
        rows=[]
        for i,x in enumerate([0,a.width,1280-a.width]):
            _,r=module.compile_one(fused_plan(),i,x,a.width);rows.append(r);print('CAPACITY_OPTIMIZED_POSITION',x,r['high'],flush=True)
        write_json(srcdir/'probe.json',dict(rows=rows,source_sha256=file_hash(source),derived_sha256=file_hash(derived),complete_frame=False));return
    assert json.loads(a.package.read_text())['plan']==plan()
    sys.argv=[sys.argv[0],'--package',str(a.package),'--name',a.name,'--width',str(a.width)];module.main()
    out=base/a.name;m=json.loads((out/'program.json').read_text());sources=[Path(__file__),Path(__file__).with_name('capacity_bank_search.py'),Path(__file__).with_name('compile_b1_capacity.py'),Path(__file__).with_name('b1_capacity_component_net.py'),Path(__file__).with_name('capacity_fixed_headers.py')]
    for f in sources:(out/'frozen'/f.name).write_bytes(f.read_bytes());m['source_hashes'][str(f)]=file_hash(f)
    m.update(capacity_variant=True,budget=budget(),node_bank_search=True,original_compiler_sha256=file_hash(source));write_json(out/'program.json',m)


if __name__=='__main__':main()
