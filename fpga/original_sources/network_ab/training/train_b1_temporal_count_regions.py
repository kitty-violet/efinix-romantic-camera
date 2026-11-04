"""Count supervision with an audited training-only temporal label revision."""
from pathlib import Path
from functools import lru_cache
import hashlib
import json
import numpy as np
import torch
from .train_b1_count_regions import CountRegionTraining,regression
from .train_b1_sixteen_regions import colour_target
from .train_b1_ordered_regions import evaluate,preview
from . import train_b1_late_region_supervision as driver
from .runtime import file_hash
from . import train_b1_minimum_area as m


@lru_cache(maxsize=32)
def revised_classes(path,digest):
    assert file_hash(path)==digest
    with np.load(path) as z:array=z['classes'].copy()
    assert array.shape==(720,1280) and array.dtype==np.uint8 and array.max()<200
    return array


class TemporalCountTraining(CountRegionTraining):
    def __init__(self,args,data):
        super().__init__(args,data)
        manifest=Path(args.class_manifest);assert manifest.is_file()
        self.class_manifest_sha256=file_hash(manifest)
        bundle=json.loads(manifest.read_text(encoding='utf-8'))
        assert bundle['schema']=='b1-temporal-class-consensus-v1'
        palette=(self.palette.detach().cpu().numpy()*128+128).astype(np.float32)
        assert bundle['palette_sha256']==hashlib.sha256(palette.tobytes()).hexdigest()
        assert bundle['original_manifest_sha256']==file_hash(m.OUT/'data.json')
        self.class_overrides=bundle['overrides']
        assert set(self.class_overrides)<=set(self.by_id)
        assert not (set(self.class_overrides)&{r['id'] for r in data['validation']+data['reserved_test']})
        for key,entry in self.class_overrides.items():
            row=self.by_id[key]
            assert entry['split']==row['split']=='train'
            assert entry['source_sha256']==row['sha256'] and entry['region_sha256']==row['region_sha256']
            revised_classes(entry['path'],entry['sha256'])

    def batch_style(self,native):
        x,target,specs,flow,valid=super().batch_style(native)
        if not hasattr(self,'class_overrides'):return x,target,specs,flow,valid
        codes=[]
        for i,spec in enumerate(specs):
            entry=self.class_overrides.get(spec['id'])
            if entry is None:codes.append(self.class_labels[i]);continue
            array=revised_classes(entry['path'],entry['sha256'])
            xx,yy=spec['xy'];w,h=spec['wh'];array=array[yy:yy+h,xx:xx+w]
            if spec['flip']:array=array[:,::-1]
            codes.append(torch.tensor(np.ascontiguousarray(array.astype(np.int64)),device=x.device))
        self.class_labels=torch.stack(codes)
        return x,colour_target(self.class_labels,self.palette),specs,flow,valid

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(temporal_class_manifest_sha256=self.class_manifest_sha256,
            temporal_training_label_revision=True,validation_labels_unchanged=True,
            inference_label_bundle_required=False)
        torch.save(state,path)

    def restore(self,path):
        state=torch.load(path,map_location='cpu',weights_only=True)
        assert state['temporal_class_manifest_sha256']==self.class_manifest_sha256
        super().restore(path)


if __name__=='__main__':
    regression();here=Path(__file__).parent
    names=['train_b1_count_regions.py','prepare_b1_temporal_classes.py','train_b1_native_distance_boundaries.py',
        'train_b1_decisive_regions.py','probe_b1_independent_decoder.py','train_b1_independent_region_boundaries.py',
        'train_b1_balanced_region_boundaries.py','train_b1_binary_regions.py','train_b1_ordered_regions.py',
        'build_b1_ordered_palette.py','minimum_area_blocks.py','train_b1_quantum_regions.py','train_b1_region_formation.py']
    driver.main(training_factory=TemporalCountTraining,extra_sources=[Path(__file__)]+[here/n for n in names],
        evaluation=evaluate,preview=preview)
