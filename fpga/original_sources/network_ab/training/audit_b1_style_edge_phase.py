"""Artificial-step phase sensitivity of actual stylizer bytes, not reconstruction.

Compare fitted boundary roughness, normal transition width and preserved step
contrast together; a blurry/constant output is not accepted for low roughness.
"""
import argparse,copy
from pathlib import Path
import numpy as np
import torch
from . import train_b1_minimum_area as m
from .audit_b1_sampling_edges import slant,edge_metrics
from .runtime import write_json,file_hash


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoints',nargs='+',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();torch.set_num_threads(2);results=[]
    for checkpoint in args.checkpoints:
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        net=m.t.r.c.ColourB1(revive=False);net.load_state_dict(state['model']);net.package['residual_encoding']=copy.deepcopy(state['residual_encoding']);net.quantized=True
        actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(net.export()),device='cpu');records=[]
        for slope in (.25,.5,1.,-.5):
            for phase in (.1,.35,.65,.9):
                rgb=slant(slope,phase);x=torch.from_numpy(rgb.transpose(2,0,1).copy())[None].float()-128
                integer=actual(x);assert torch.equal(integer,net(x/128)[0]*128)
                out=integer[0].permute(1,2,0).numpy()+128
                try:metrics=edge_metrics(out,slope,phase)
                except AssertionError:metrics=dict(invalid_step_response=True)
                records.append(dict(slope=slope,phase=phase,**metrics))
        valid=[r for r in records if not r.get('invalid_step_response')]
        mean={k:float(np.mean([r[k] for r in valid])) for k in ('contour_residual_rms','transition_width_normal_median','contrast_ratio')} if valid else {}
        results.append(dict(checkpoint=checkpoint,sha256=file_hash(checkpoint),records=records,mean=mean,invalid_count=len(records)-len(valid)))
        print('STYLE_STEP_RESPONSE',Path(checkpoint).parent.name,mean,flush=True)
    write_json(args.out,dict(results=results,actual_integer_bytes=True,scope='Artificial colour-step diagnostic. Not style-image or video acceptance.'))


if __name__=='__main__':main()
