"""Rebind weights after QAT only if graph/scales/header offsets are identical."""
import argparse,copy,json,shutil
from pathlib import Path
from .train_luma_regions import ROOT,RUN
from .region_fixed_headers import coefficient_image
from .runtime import file_hash,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--program',type=Path,required=True);p.add_argument('--package',type=Path,required=True);p.add_argument('--name',required=True);a=p.parse_args();old=json.loads((a.program/'program.json').read_text());assert old['status']=='C3_ROW_PAIR_COMPILER_ONLY_PASS'
    before_path=Path(old['package']);assert file_hash(before_path)==old['package_sha256'];before=json.loads(before_path.read_text());after=json.loads(a.package.read_text())
    for key in ['schema','plan','scales','residual','levels']:assert before[key]==after[key],key
    assert before['parameters'].keys()==after['parameters'].keys()
    for n in before['parameters']:
        for k in ['input_scale','output_scale']:assert before['parameters'][n][k]==after['parameters'][n][k]
    out=ROOT/'runs'/RUN/a.name;out.mkdir(exist_ok=False);coefficient_image(after,out)
    assert json.loads((a.program/'coefficient_map.json').read_text())['operators']==json.loads((out/'coefficient_map.json').read_text())['operators']
    for name in ['compute.bin','compute.hex','dma.bin','dma.hex','commands.json.gz','fields.hex']:shutil.copy2(a.program/name,out/name);assert file_hash(a.program/name)==file_hash(out/name)
    for s in old['streams']:assert file_hash(out/(s['name']+'.bin'))==s['sha256']
    frozen=out/'frozen';frozen.mkdir()
    for name,h in old['source_hashes'].items():
        src=a.program/'frozen'/Path(name).name;assert file_hash(src)==h;shutil.copy2(src,frozen/src.name)
    f=Path(__file__);shutil.copy2(f,frozen/f.name);result=copy.deepcopy(old);result.update(package=str(a.package),package_sha256=file_hash(a.package),rebind=dict(previous_program=str(a.program),previous_manifest_sha256=file_hash(a.program/'program.json'),descriptors_identical=True,graph_scales_and_abi_identical=True,weights_changed=True,needs_new_numeric_test=True));result['source_hashes'][str(f)]=file_hash(f)
    write_json(out/'program.json',result);print('C3_ROW_PAIR_WEIGHTS_REBOUND',old['commands'],flush=True)


if __name__=='__main__':main()
