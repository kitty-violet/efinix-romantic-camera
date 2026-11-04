"""Accumulate all seven TRAIN scenes before one update; unchanged integer B1.

This is an optimization diagnostic. It is not video/generalization acceptance.
No labels, colour cleanup, history or auxiliary decoder enter inference.
"""
from pathlib import Path
import time
import torch
from .train_b1_seven_scene_fit import SevenSceneFitTraining
from .train_b1_count_regions import count_targets,count_loss,regression
from .train_b1_temporal_count_regions import revised_classes
from .fit_b1_sixteen_regions import region_class_map
from . import train_b1_minimum_area as m
from .train_b1_ordered_regions import evaluate,preview
from . import train_b1_late_region_supervision as driver


class SevenSceneBatchTraining(SevenSceneFitTraining):
    preflight_native_resume=True

    def __init__(self,args,data):
        super().__init__(args,data)
        self.fit_cache=[]
        for key in self.fit_ids:
            row=self.by_id[key]
            x=m.t.r.tensor(m.t.r.read(row['path']))[None]
            entry=self.class_overrides.get(key)
            codes=revised_classes(entry['path'],entry['sha256']) if entry else region_class_map(row['target'],row['region'],self.pk)
            truth=count_targets(torch.tensor(codes.astype('int64'),device=x.device)[None])
            self.fit_cache.append((key,x,truth))

    def update(self,native=None):
        if native is False:return super().update(False)
        began=time.perf_counter();self.opt.zero_grad(set_to_none=True)
        parameters=[p for p in self.net.parameters() if p.requires_grad]
        losses=[];gradients=[];previous=None
        for key,x,truth in self.fit_cache:
            nodes=self.net(x,all_nodes=True)
            logits=(nodes['up1_high_dw'][:,:15]-63.5)/4
            loss=count_loss(logits,truth)
            (loss/len(self.fit_cache)).backward()
            losses.append(float(loss.detach()))
            if self.step==0:
                accumulated=torch.cat([p.grad.detach().flatten() for p in parameters])
                gradients.append(accumulated if previous is None else accumulated-previous)
                previous=accumulated
            del nodes,logits,loss
        grad=torch.nn.utils.clip_grad_norm_(parameters,5,error_if_nonfinite=True)
        assert grad>0 and all(torch.isfinite(p.grad).all() for p in parameters)
        for group,lr in zip(self.opt.param_groups,self.base_lrs):group['lr']=lr
        self.opt.step()
        with torch.no_grad():
            for key,w in self.net.weight.items():
                if w.requires_grad:
                    limit=127*(getattr(self.net,key+'_m')/131072).float().flatten()[:,None,None,None]
                    w.copy_(w.maximum(-limit).minimum(limit))
            for key,(w,b) in self.fixed.items():
                assert torch.equal(w,self.net.weight[key]) and torch.equal(b,self.net.bias[key])
        self.step+=1;torch.cuda.synchronize()
        result=dict(step=self.step,loss=sum(losses)/len(losses),values=dict(seven_scene_count_ce=sum(losses)/len(losses)),
            per_scene_losses=dict(zip(self.fit_ids,losses)),grad=float(grad),native720=True,
            samples=[dict(id=key,xy=[0,0],wh=[1280,720],flip=False) for key in self.fit_ids],
            temporal_pair=False,accumulated_scenes=7,optimizer_updates=1,seconds=time.perf_counter()-began)
        if gradients:
            g=torch.stack(gradients);g=g/g.norm(dim=1,keepdim=True).clamp_min(1e-12)
            result['initial_scene_gradient_cosine']=(g@g.T).cpu().tolist()
            print('SEVEN_SCENE_GRADIENT_COSINE',result['initial_scene_gradient_cosine'],flush=True)
        return result

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(seven_scene_gradient_accumulation=True,
            native_coverage_schedule='all seven full720 TRAIN frames averaged before each optimizer update',
            native_objective='mean of seven independently class-balanced count cross-entropies',
            generalization_claim=False)
        torch.save(state,path)


if __name__=='__main__':
    regression();here=Path(__file__).parent
    names=['train_b1_seven_scene_fit.py','train_b1_native_coverage_counts.py','train_b1_truth_area_counts.py',
        'train_b1_temporal_count_regions.py','train_b1_count_regions.py','prepare_b1_temporal_classes.py',
        'train_b1_native_distance_boundaries.py','train_b1_decisive_regions.py','probe_b1_independent_decoder.py',
        'train_b1_independent_region_boundaries.py','train_b1_balanced_region_boundaries.py','train_b1_binary_regions.py',
        'train_b1_ordered_regions.py','build_b1_ordered_palette.py','minimum_area_blocks.py','train_b1_quantum_regions.py',
        'train_b1_region_formation.py']
    driver.main(training_factory=SevenSceneBatchTraining,extra_sources=[Path(__file__)]+[here/n for n in names],
        evaluation=evaluate,preview=preview)
