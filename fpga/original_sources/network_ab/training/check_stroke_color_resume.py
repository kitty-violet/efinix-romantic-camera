"""Exercise restoration with active temporal/phase losses, separately from training."""
import copy
import json
import torch
from . import train_stroke_color as base
from .stroke_color_input_net import StrokeColorInputNet
from .train_stroke_color_input import Training
from .runtime import configure_workspace,contract_guard,write_json
from .engine import set_reproducible


def main():
    configure_workspace();contract_guard();set_reproducible(2026100487);torch.set_num_threads(4)
    base.StrokeColorNet=StrokeColorInputNet
    out=base.ROOT/'native-stroke-color-input-v2';t=Training(out)
    if not (out/'eval-000000').exists():t.evaluate()
    # Set the diagnostic schedule to 500 so temporal losses are fully enabled.
    # These are disposable updates, NOT 500 training steps.
    t.step=500;t.update(native=False)
    torch.save(t.state(),out/'active_temporal_preflight.pt')
    ref=t.update(native=False);expected=copy.deepcopy(t.net.state_dict())
    t.restore(torch.load(out/'active_temporal_preflight.pt',map_location='cpu',weights_only=True));replay=t.update(native=False)
    error=max(float((v-expected[k]).abs().max()) for k,v in t.net.state_dict().items())
    passed=ref['sample']==replay['sample'] and error<=1e-6
    write_json(out/'active_temporal_resume.json',dict(status='PASS' if passed else 'FAIL',resume_max_error=error,reference=ref,replay=replay,
        disposable_gate_updates=True,schedule_step_set_to_500_for_temporal_gate=True,actual_formal_updates=0,
        determinism_note='CUDA grid_sample backward warns of possible nondeterminism; this replay is measured, not a universal determinism guarantee'))
    print('ACTIVE_TEMPORAL_RESUME',passed,error,flush=True)
    assert passed


if __name__=='__main__':main()
