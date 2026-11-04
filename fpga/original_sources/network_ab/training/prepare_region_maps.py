"""Small paired synthesis projects in the model-training workspace."""
from pathlib import Path
import shutil
from .region_detail_target import ROOT,RUN
from .runtime import file_hash,write_json
from .validate_region_native import NATIVE

def project(root,name,files):
    root.mkdir(exist_ok=False);records=[]
    for f in files:
        dst=root/f.name;shutil.copy2(f,dst);records.append(dict(path=str(f),source_sha256=file_hash(f),copied_sha256=file_hash(dst)))
    template=(ROOT.parent/'tmp/frame-execution/arena-map1/tensor_arena12.xml').read_text()
    template=template.replace('tensor_arena12',name)
    old=f'<efx:design_file name="{name}.sv" version="default" library="default" />'
    new='\n'.join(f'<efx:design_file name="{f.name}" version="default" library="default" />' for f in files)
    assert old in template;template=template.replace(old,new)
    (root/(name+'.xml')).write_text(template)
    (root/(name+'.sdc')).write_text('create_clock -name clk -period 6.666 [get_ports clk]\n')
    write_json(root/'inputs.json',dict(top=name,files=records,isolated=True,board_fit=False,timing=False))

def main():
    root=ROOT/'runs'/RUN;dst=root/'rtl-variable-groups/frozen';ref=NATIVE/'frozen'
    for name,src in [('map-fuse-reference',ref),('map-fuse-variable',dst)]:
        project(root/name,'pw_skip_fuse4',[src/'pw_skip_fuse4.sv',src/'shallow_result_fifo.sv'])

if __name__=='__main__':main()
