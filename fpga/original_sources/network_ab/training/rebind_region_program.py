"""Reuse immutable descriptors only when graph, scales and ABI stay identical."""
import argparse,copy,json,shutil
from pathlib import Path
from .region_detail_target import ROOT,RUN
from .region_fixed_headers import coefficient_image
from .runtime import file_hash,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--program',type=Path,required=True);p.add_argument('--package',type=Path,required=True);p.add_argument('--name',required=True);a=p.parse_args()
    old=json.loads((a.program/'program.json').read_text());assert old['fixed_header_windows']
    oldp=Path(old['package']);assert file_hash(oldp)==old['package_sha256']
    before=json.loads(oldp.read_text());after=json.loads(a.package.read_text())
    for key in ['plan','scales','residual','levels']:assert before[key]==after[key],key
    assert before['parameters'].keys()==after['parameters'].keys()
    for key in before['parameters']:
        for field in ['input_scale','output_scale']:assert before['parameters'][key][field]==after['parameters'][key][field],(key,field)
    root=ROOT/'runs'/RUN/a.name;root.mkdir(exist_ok=False);ops=coefficient_image(after,root)
    oldmap=json.loads((a.program/'coefficient_map.json').read_text());newmap=json.loads((root/'coefficient_map.json').read_text())
    assert oldmap['operators']==newmap['operators']
    for item in old['streams']:
        for suffix in ['.bin','.hex']:shutil.copy2(a.program/(item['name']+suffix),root/(item['name']+suffix))
        assert file_hash(root/(item['name']+'.bin'))==item['sha256']
    frozen=root/'frozen_sources';frozen.mkdir()
    for path,h in old['source_hashes'].items():
        src=a.program/'frozen_sources'/Path(path).name
        assert src.exists() and file_hash(src)==h,(src,h)
        shutil.copy2(src,frozen/src.name)
    f=Path(__file__);shutil.copy2(f,frozen/f.name)
    result=copy.deepcopy(old);result.update(package=str(a.package),package_sha256=file_hash(a.package),
        rebind=dict(previous_program=str(a.program),previous_program_sha256=file_hash(a.program/'program.json'),
                    weights_changed=True,descriptors_identical=True,graph_scales_residual_abi_identical=True,requires_new_numeric_verification=True))
    result['source_hashes'][str(f)]=file_hash(f);write_json(root/'program.json',result)
    print('REGION_PROGRAM_REBOUND',result['commands'],flush=True)

if __name__=='__main__':main()
