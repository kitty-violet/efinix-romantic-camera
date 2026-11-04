"""Keep native region luminance labels invariant to cropping and flow."""
from functools import lru_cache
from pathlib import Path
import numpy as np,torch
from .train_b1_rendered_luma_readout import RenderedLumaReadoutTraining,driver,preview
from .train_b1_real_region_texture import align_pair
from .train_b1_reference_profile_distance import evaluate as rgb_evaluate
from .train_b1_ordered_distance_rims import distance_objective
from .b1_joint_interface_targets import joint_interface_fields
from .audit_b1_rim_tolerance import rim_terms
from .runtime import file_hash
from . import train_b1_minimum_area as m


@lru_cache(maxsize=16)
def native_codes(path):
    with np.load(path) as z:return z['codes'].astype(np.int64)


class NativeLumaReadoutTraining(RenderedLumaReadoutTraining):
    def __init__(self,args,data):
        assert data.get('native_luma_definition')
        for row in data['training']+data['validation']:
            assert file_hash(row['native_luma_codes'])==row['native_luma_codes_sha256']
            assert file_hash(row['hard_region_target'])==row['native_luma_colour_source_sha256']
        super().__init__(args,data)

    def batch_style(self,native):
        x,t,specs,flow,valid=m.AreaTraining.batch_style(self,native);codes=[]
        for i,spec in enumerate(specs):
            row=self.by_id.get(spec['id']) or self.by_stem[spec['id']];xx,yy=spec['xy'];w,h=spec['wh']
            c=native_codes(row['native_luma_codes'])[yy:yy+h,xx:xx+w]
            source=m.t.r.read(row['path'])[yy:yy+h,xx:xx+w]
            if spec['flip']:c=c[:,::-1];source=source[:,::-1]
            codes.append(torch.from_numpy(c.copy()).to(x.device))
            expected=torch.from_numpy(source.transpose(2,0,1).copy()).to(x.device).float()/128-1
            assert torch.equal(x[i],expected)
        self.class_labels=torch.stack(codes);self.motion_fraction=x.new_zeros(())
        if flow is not None:
            t,self.labels,self.class_labels,self.motion_fraction=align_pair(x,t,self.labels,self.class_labels,flow,valid)
        elif self.step%4 in (1,2):self.consume_contrast_draws(self.labels)
        return x,t,specs,flow,valid

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(native_region_luma_labels=True,luminance_label='verified native flat region colours BEFORE antialias/crop/flow',
            crop_independent_luma_labels=True,flow_transports_luma_codes_directly=True,
            original_rgb_target_unchanged=True,visual_acceptance=False)
        torch.save(s,path)


@torch.inference_mode()
def evaluate(net,rows):
    result=rgb_evaluate(net,rows)
    for rec,row in zip(result['records'],rows):
        x=m.t.r.tensor(m.t.r.read(row['path']))[None];t=m.t.r.tensor(m.t.r.read(row['target']))[None]
        lab=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None].to(x.device)
        code=torch.from_numpy(native_codes(row['native_luma_codes']))[None].to(x.device)
        truth=joint_interface_fields(code)[:,:14];nodes=net(x,all_nodes=True)
        full,band=distance_objective(nodes['up1_low_pw'][:,:14],truth);rim,_=rim_terms(nodes['rgb_head']/128,t,lab,2)
        rec.update(luminance_interface_full=float(full),luminance_interface_band=float(band),two_pixel_region_centred_bright_shoulder=float(rim))
    for key in ('luminance_interface_full','luminance_interface_band','two_pixel_region_centred_bright_shoulder'):
        result['mean'][key]=sum(r[key] for r in result['records'])/len(rows)
    return result


if __name__=='__main__':
    torch.set_num_threads(2)
    driver.main(training_factory=NativeLumaReadoutTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
