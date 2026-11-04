"""Train final RGB geometry on all source videos with the fixed B1 graph.

No hard luminance labels, auxiliary decoders, or latent-code objectives.
Whole native720 frames and reliable adjacent source-frame pairs are retained.
"""
import copy
from pathlib import Path
import numpy as np,torch
from . import train_b1_minimum_area as m
from .train_b1_complete_rgb_geometry import CompleteRGBGeometryTraining
from .train_b1_trainable_luma_readout import ReadoutB1
from .b1_fine_spatial_decoder import install
from .b1_refine_live_luma import refine
from .whitened_b1_optimizer import patch_statistics,WhitenedB1Adam
from .train_b1_clean_regions import CleanRegionTraining
from .train_b1_real_region_texture import align_pair
from .train_b1_native_coverage_counts import planned_native
from .train_b1_late_region_supervision import main as driver


class FineRGBNativeTraining(CompleteRGBGeometryTraining,m.AreaTraining):
    preflight_native_resume=True
    input_stability_loss=CleanRegionTraining.input_stability_loss

    @staticmethod
    def appearance_loss(p,t,mode):return p.sum()*0,{}

    def additional_loss(self,p,t,native):
        if hasattr(self,'rgb_objective'):
            labels=self.labels if native else self.labels[...,40:-40,40:-40]
            return self.rgb_objective(p,t,labels)
        return CompleteRGBGeometryTraining.additional_loss(self,p,t,native)

    def __init__(self,args,data):
        assert args.qat and not args.freeze_head and args.aux==0 and args.scale_lr==0
        m.AreaTraining.__init__(self,args,data)
        state=torch.load(args.warm,map_location='cpu',weights_only=True)
        self.net.__class__=ReadoutB1
        self.calibration_ids=[];crops=[]
        for video in sorted({r['video'] for r in data['training']}):
            rows=[r for r in data['training'] if r['video']==video];row=rows[len(rows)//2]
            self.calibration_ids.append(row['id']);crops.append(m.t.r.tensor(m.t.r.read(row['path'])[200:520,480:800]))
        x=torch.stack(crops)
        with torch.no_grad():
            before=self.net(x,all_nodes=True)
            if not state.get('fine_full_decoder_trainable'):install(self.net)
            if int(self.net.up2_low_pw_m[0,0,0,0])==8257:refine(self.net)
            after=self.net(x,all_nodes=True)
            assert torch.equal(before['rgb_head'],after['rgb_head'])
            assert all(torch.equal(before[k],v) for k,v in after.items() if k not in ('up2_low_pw','up2_nn','up2_high_dw'))
            del before,after
        groups=[]
        for k,w in self.net.weight.items():
            w.requires_grad_(True);self.net.bias[k].requires_grad_(True)
            s=(getattr(self.net,k+'_m')/131072).float().flatten()
            groups.extend([dict(params=[w],lr=args.lr,row_gain=(s*256)[:,None,None,None]),dict(params=[self.net.bias[k]],lr=.004)])
        raw=self.net.export();stats=patch_statistics(self.net,x,4096);assert self.net.export()==raw
        self.opt=WhitenedB1Adam(groups,self.net,stats,True);self.base_lrs=[g['lr'] for g in self.opt.param_groups]
        self.initial_export=raw

    def batch_style(self,native):
        x,t,specs,flow,valid=m.AreaTraining.batch_style(self,native)
        for i,spec in enumerate(specs):
            row=self.by_id.get(spec['id']) or self.by_stem[spec['id']];xx,yy=spec['xy'];w,h=spec['wh']
            source=m.t.r.read(row['path'])[yy:yy+h,xx:xx+w]
            if spec['flip']:source=source[:,::-1]
            expected=torch.from_numpy(source.transpose(2,0,1).copy()).to(x.device).float()/128-1
            assert torch.equal(x[i],expected)
        if flow is not None:
            t,self.labels,_,fraction=align_pair(x,t,self.labels,self.labels,flow,valid)
            for spec in specs:spec['source_flow_target_fraction']=float(fraction)
        self.last_source=x.detach()
        return x,t,specs,flow,valid

    def update(self,native=None):
        return m.AreaTraining.update(self,planned_native(self.step) if native is None else native)

    def preflight_extras(self):
        while self.step%4:self.update(True)
        result=self.update(False)
        assert result['temporal_pair'] and 'tiny_input_stability' in result['values'] and 'complete_rgb_profile' in result['values']
        with torch.no_grad():
            nodes=self.net(self.last_source,all_nodes=True)
            rt=m.t.r.IntegerB1(m.t.r.runtime_from_image(self.net.export()),device='cuda')
            actual,_=rt(self.net.round(self.last_source*128),True)
            assert all(torch.equal(v,actual[k]) for k,v in nodes.items())
        return dict(update=result,actual_all_nodes_error=0,bytes=len(self.net.export()),training_input='exact source crop or flip')

    def save(self,path):
        m.AreaTraining.save(self,path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(qat=True,fine_full_decoder_trainable=True,live_luma_precision_refined=True,
            final_rgb_training=True,fixed_palette=False,latent_geometry_supervision=False,local_fit_only=False,
            geometry_profile='all target RGB profiles including thin strips, shared offset +/-2px',
            calibration_ids=self.calibration_ids,whole_frame_schedule='3 of 4 native720, 1 of 4 actual flow pair crop',
            extra_inference_operations=0,visual_acceptance=False)
        torch.save(state,path)


@torch.no_grad()
def evaluate(net,rows):
    records=[]
    for row in rows:
        x=m.t.r.tensor(m.t.r.read(row['path']))[None];t=m.t.r.tensor(m.t.r.read(row['target']))[None]
        p=net(x)[0];obj=object.__new__(CompleteRGBGeometryTraining)
        obj.labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None].to(x.device)
        loss,values=obj.additional_loss(p,t,True);d=(p-t)*128
        records.append(dict(id=row['id'],score=float(loss),mae=float(d.abs().mean()),
            bias_Co=float(((d[:,0]-d[:,2])/2).mean()),bias_Cg=float(((2*d[:,1]-d[:,0]-d[:,2])/4).mean()),
            **{k:float(v) for k,v in values.items()}))
    return dict(records=records,mean={k:sum(r[k] for r in records)/len(records) for k in records[0] if k!='id'})


if __name__=='__main__':
    torch.set_num_threads(2)
    driver(training_factory=FineRGBNativeTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=m.t.preview)
