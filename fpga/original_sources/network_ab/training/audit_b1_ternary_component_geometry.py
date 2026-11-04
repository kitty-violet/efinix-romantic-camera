"""Compare exact count-code region geometry; low region count alone is not quality."""
import argparse,copy
from pathlib import Path
import numpy as np
import torch
from . import train_b1_minimum_area as m
from .train_b1_projected_count_regions import region_geometry
from .runtime import write_json,file_hash


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoints',nargs='+',required=True)
    ap.add_argument('--data',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();torch.set_num_threads(2);data=m.load_data(a.data);results=[]
    for checkpoint in a.checkpoints:
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        net=m.t.r.c.ColourB1(revive=False);net.load_state_dict(state['model'])
        net.package['residual_encoding']=copy.deepcopy(state['residual_encoding']);net.quantized=True
        actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(net.export()),device='cpu')
        rows=[]
        for row in data['validation']:
            rgb=m.t.r.read(row['path'])
            x=torch.from_numpy(rgb.transpose(2,0,1).copy())[None].float()-128
            nodes=net(x/128,all_nodes=True)
            assert torch.equal(actual(x),nodes['rgb_head'])
            z=nodes['up2_low_dw'][0,:15]
            code=(81*z[:7].sum(0)+9*z[7:11].sum(0)+z[11:15].sum(0)).long().numpy()
            rows.append(dict(id=row['id'],**region_geometry(code)))
        means={key:float(np.mean([r[key] for r in rows])) for key in rows[0] if key!='id'}
        results.append(dict(checkpoint=checkpoint,sha256=file_hash(checkpoint),records=rows,mean=means))
        print('TERNARY_COMPONENT_GEOMETRY',Path(checkpoint).parent.name,means,flush=True)
    write_json(a.out,dict(actual_byte_rgb_equal=True,results=results,
        scope='Count codes at360p; tiny regions and categorical vote changes are diagnostics, not visual acceptance.'))


if __name__=='__main__':main()
