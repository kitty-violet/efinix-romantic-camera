"""Use native720 frames for supervised region learning; retain flow crops.

The former schedule had one full frame per150 updates. With crop320 and40px
loss margins, most updates supervise only2 or3 small240x240 areas. Here three
of every four updates supervise the whole native frame, and the fourth keeps
the existing adjacent-frame crop/area/noise/phase objectives. B1 is unchanged.
"""
from pathlib import Path
import hashlib
import torch
from .train_b1_truth_area_counts import TruthAreaCountTraining,regression
from .train_b1_count_regions import regression as count_regression
from .train_b1_ordered_regions import evaluate,preview
from . import train_b1_late_region_supervision as driver


def planned_native(step):return step%4!=0


class NativeCoverageCountTraining(TruthAreaCountTraining):
    def __init__(self,args,data):
        super().__init__(args,data)
        warm=torch.load(args.warm,map_location='cpu',weights_only=True)
        self.packed_decoder=warm.get('packed_native_count_decoder')
        self.group_decoder=warm.get('group_native_count_decoder')
        assert not (self.packed_decoder and self.group_decoder)
        if self.packed_decoder or self.group_decoder:
            assert args.qat,'Native720 rounding is an integer-only experiment'
            assert (self.packed_decoder or self.group_decoder) in ('box','gauss')
        # Also proves FP32->QAT did not silently recalibrate the fixed program.
        for key in self.net.weight:
            assert torch.equal(getattr(self.net,key+'_m').cpu(),warm['model'][key+'_m'])
        self.initial_integer_sha256=hashlib.sha256(self.net.export()).hexdigest()
        if self.packed_decoder or self.group_decoder:
            trained=warm.get('group_decoder_training') or warm.get('packed_decoder_training')
            if not trained:
                image=Path(args.warm).parent/'coefficients-padded.bin'
                assert image.is_file() and hashlib.sha256(image.read_bytes()).hexdigest()==self.initial_integer_sha256
            else:assert warm.get('qat') or warm.get('args',{}).get('qat')

    def update(self,native=None):
        native=planned_native(self.step) if native is None else native
        result=super().update(native)
        result['supervised_native_pixels']=sum(w*h if native else (w-80)*(h-80)
            for w,h in (s['wh'] for s in result['samples']))
        return result

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(native_coverage_schedule='step%4: 0 existing flow crops; 1,2,3 native720',
            architecture_changed=False,extra_inference_operations=0,initial_integer_sha256=self.initial_integer_sha256)
        if self.packed_decoder:state.update(packed_native_count_decoder=self.packed_decoder,
            training_proxy_not_yet_verified=False,packed_decoder_training=True)
        if self.group_decoder:state.update(group_native_count_decoder=self.group_decoder,
            training_proxy_not_yet_verified=False,group_decoder_training=True)
        torch.save(state,path)


def schedule_regression():
    flags=[planned_native(i) for i in range(32)]
    assert sum(flags)==24 and all(flags[i]==(i%4!=0) for i in range(32))
    # Resume is a function of the saved global step, with no hidden counter.
    assert [planned_native(i) for i in range(11,32)]==flags[11:]
    assert all(i%4==0 for i,f in enumerate(flags) if not f)
    print('NATIVE_COVERAGE_SCHEDULE_REGRESSION_PASS',flush=True)


if __name__=='__main__':
    count_regression();regression();schedule_regression();here=Path(__file__).parent
    names=['train_b1_truth_area_counts.py','train_b1_temporal_count_regions.py','train_b1_count_regions.py',
        'prepare_b1_temporal_classes.py','train_b1_native_distance_boundaries.py','train_b1_decisive_regions.py',
        'probe_b1_independent_decoder.py','train_b1_independent_region_boundaries.py','train_b1_balanced_region_boundaries.py',
        'train_b1_binary_regions.py','train_b1_ordered_regions.py','build_b1_ordered_palette.py','minimum_area_blocks.py',
        'train_b1_quantum_regions.py','train_b1_region_formation.py']
    driver.main(training_factory=NativeCoverageCountTraining,extra_sources=[Path(__file__)]+[here/n for n in names],
        evaluation=evaluate,preview=preview)
