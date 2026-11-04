"""Measure the known legacy teacher truncation versus diagnostic rounding."""
import json
import numpy as np
import torch
from .audit_style_branches import RUN,load,predict,spatial
from .train_b1_distillation import as_tensor,read_image
from .smooth_surface_guard import correct_surface
from .runtime import configure_workspace,write_json
from .engine import set_reproducible


def main():
    root=configure_workspace();set_reproducible(2026100401);run=root/'runs'/RUN;rows=json.loads((run/'assets.json').read_text())['rows']
    net=load(root,'teacher');results=[];torch.set_num_threads(2)
    for row in rows:
        if row['split']=='continuous_diagnostic':continue
        s=read_image(row['source']);t=read_image(row['target'])
        with torch.inference_mode():raw=(net(as_tensor(s)[None].cuda())[0].permute(1,2,0).cpu().numpy()*128+128).clip(0,255).astype(np.uint8)
        legacy=correct_surface(s,raw);rounded=predict(net,s,'teacher')
        results.append(dict(id=row['id'],legacy_target_max_error=int(np.abs(legacy.astype(int)-t.astype(int)).max()),
            round_to_legacy_mae=float(np.abs(rounded.astype(float)-legacy).mean()),round_to_legacy_max=int(np.abs(rounded.astype(int)-legacy.astype(int)).max())))
    assert max(r['legacy_target_max_error'] for r in results)==0
    write_json(run/'quantization_note.json',dict(status='PASS_LEGACY_TARGET_REPRODUCED',records=results,
        interpretation='Audit teacher uses rounding before R48; legacy20 static targets use truncation. Continuous diagnostic targets use rounding consistently. All interventions compared to their own unchanged baseline. Static baseline has about0.5/255 offset, not a model error. Do not interpret sub-quantization static changes as meaningful. Any new training retains exact legacy target files.'))
    print('QUANTIZATION_VERIFIED',flush=True)


if __name__=='__main__':main()
