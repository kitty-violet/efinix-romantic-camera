"""Controlled fit of seven complete TRAIN scenes before scaling coverage.

One representative middle frame from each training source. No validation or
reserved scene is fitted. Diagnostics must distinguish this from generalization.
"""
from pathlib import Path
import torch
from .train_b1_native_coverage_counts import NativeCoverageCountTraining
from .train_b1_truth_area_counts import regression as area_regression
from .train_b1_count_regions import regression as count_regression
from .train_b1_count_regions import count_targets
from .fit_b1_sixteen_regions import region_class_map
from .train_b1_temporal_count_regions import revised_classes
from . import train_b1_minimum_area as m
from .train_b1_ordered_regions import evaluate,preview
from . import train_b1_late_region_supervision as driver


class SevenSceneFitTraining(NativeCoverageCountTraining):
    def __init__(self,args,data):
        super().__init__(args,data)
        assert args.shift==0 and args.temporal==0 and args.lr_schedule=='constant'
        selected={}
        for video,rows in sorted(self.groups.items()):
            rows=sorted(rows,key=lambda r:r['id']);selected[video]=[rows[len(rows)//2]]
        assert len(selected)==7 and all(r['split']=='train' for rows in selected.values() for r in rows)
        self.groups=selected;self.fit_ids=[rows[0]['id'] for rows in selected.values()]

    def update(self,native=None):
        return super().update(True if native is None else native)

    @torch.inference_mode()
    def report_training_fit(self):
        records=[]
        for key in self.fit_ids:
            row=self.by_id[key];x=m.t.r.tensor(m.t.r.read(row['path']))[None]
            nodes=self.net(x,all_nodes=True);binary=nodes['up2_low_dw'][0,:15]>0
            actual=binary[:7].sum(0)*25+binary[7:11].sum(0)*5+binary[11:].sum(0)
            entry=self.class_overrides.get(key)
            codes=revised_classes(entry['path'],entry['sha256']) if entry else region_class_map(row['target'],row['region'],self.pk)
            truth=count_targets(torch.tensor(codes.astype('int64'),device=x.device)[None])
            pure=torch.stack([v.amax(1)[0]>.999 for v in truth]).all(0)
            wanted=sum(v.argmax(1)[0]*k for v,k in zip(truth,(25,5,1)))
            records.append(dict(id=key,actual_count_accuracy_pure=float((actual[pure]==wanted[pure]).float().mean())))
        result=dict(records=records,mean_accuracy=sum(r['actual_count_accuracy_pure'] for r in records)/len(records),
            training_fit_only=True,heldout=False)
        print('SEVEN_SCENE_TRAIN_FIT',self.step,result,flush=True)
        return result

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(seven_scene_fit_diagnostic=True,fit_ids=self.fit_ids,
            native_coverage_schedule='all updates full720; seven TRAIN source-middle frames',
            generalization_claim=False)
        torch.save(state,path)


if __name__=='__main__':
    area_regression();count_regression();here=Path(__file__).parent
    names=['train_b1_native_coverage_counts.py','train_b1_truth_area_counts.py','train_b1_temporal_count_regions.py',
        'train_b1_count_regions.py','prepare_b1_temporal_classes.py','train_b1_native_distance_boundaries.py',
        'train_b1_decisive_regions.py','probe_b1_independent_decoder.py','train_b1_independent_region_boundaries.py',
        'train_b1_balanced_region_boundaries.py','train_b1_binary_regions.py','train_b1_ordered_regions.py',
        'build_b1_ordered_palette.py','minimum_area_blocks.py','train_b1_quantum_regions.py','train_b1_region_formation.py']
    driver.main(training_factory=SevenSceneFitTraining,extra_sources=[Path(__file__)]+[here/n for n in names],
        evaluation=evaluate,preview=preview)
