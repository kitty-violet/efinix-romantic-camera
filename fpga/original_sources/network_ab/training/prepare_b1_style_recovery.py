"""Freeze accepted gentle-v3 targets; no change to deployed B1 operators."""
import copy
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import train_b1_reconstruction as r
from .runtime import configure_workspace, contract_guard, file_hash, write_json

OUT = r.OUT.parent/'b1-style-recovery-20261005'
PREVIOUS = r.OUT.parent/'b1-deploy-preview-20261005-v1'


@torch.inference_mode()
def main():
    configure_workspace(); contract_guard()
    from .publish_global_blocks import load, bindings
    from .publish_coarse_people import probability
    from .coarse_people import refined_mask, composite
    from .coarse_people_blocks import person_surface
    from .gentle_block_boundaries import process
    from .refine_block_separation import PARAMETERS
    OUT.mkdir(exist_ok=True); (OUT/'targets').mkdir(exist_ok=True)
    old = json.loads((PREVIOUS/'data.json').read_text(encoding='utf-8'))
    old_binding = json.loads((PREVIOUS/'binding.json').read_text())
    assert file_hash(PREVIOUS/'data.json') == old_binding['data_sha256']
    assert all(file_hash(p)==h for p,h in old_binding['code'].items())
    train,val = r.data(); train={v['sha256']:v for v in train}
    known={}
    for pair in old['pairs']:
        for key in ('current','next'):
            sha=pair[key+'_sha256']; path=pair[key]
            known[sha]=(pair[key+'_target'],pair[key+'_target_sha256'])
            train[sha]=dict(id=Path(path).stem,path=path,sha256=sha,video=pair['source'],split='train')
    for row in old['validation']:
        known[row['source_sha256']] = (row['target'],row['target_sha256'])
    test=json.loads((r.OUT.parent/'b1-reconstruction-research-20261005/holdout/manifest.json').read_text())['records']
    assert not {v['video'] for v in train.values()} & {v['video'] for v in val+test}
    assert not {v['video'] for v in val} & {v['video'] for v in test}
    cfg,net,b,d,p,adapter,clip,state,seg=load()
    def generate(x):
        pred=np.array(r.c.rgb_image(net(r.tensor(x)[None])[0]))
        surface=d.surface(p.tone(pred,adapter),d.PRESETS['direction4'])
        background=b.transform_surface(surface,'balanced')
        mask=refined_mask(probability(seg,x),x)
        people=person_surface(pred,adapter,p,'strongblocks',x)
        return process(composite(background,people,mask),PARAMETERS)
    # Reproduce one previously frozen output before producing more targets.
    sample=val[0]; expected=r.read(known[sample['sha256']][0])
    got=generate(r.read(sample['path']))
    error=int(np.abs(got.astype(np.int16)-expected.astype(np.int16)).max())
    assert error==0, ('recipe replay changed',error)
    write_json(OUT/'target_replay.json',dict(id=sample['id'],maximum_pixel_error=error,
        source_sha256=sample['sha256'],target_sha256=known[sample['sha256']][1]))
    rows=list(train.values())+val+test
    for i,row in enumerate(rows):
        started=time.perf_counter(); assert file_hash(row['path'])==row['sha256']
        if row['sha256'] in known:
            path,sha=known[row['sha256']]; assert file_hash(path)==sha
        else:
            path=OUT/'targets'/(row['sha256'][:20]+'.png')
            if not path.exists(): Image.fromarray(generate(r.read(row['path']))).save(path)
            sha=file_hash(path)
        row.update(target=str(path),target_sha256=sha,target_same_as_input=False)
        print('STYLE_TARGET',i+1,len(rows),round(time.perf_counter()-started,2),flush=True)
    files=bindings(cfg,state)
    for name in ('prepare_b1_style_recovery.py','publish_global_blocks.py','publish_coarse_people.py',
                 'coarse_people.py','coarse_people_blocks.py','gentle_block_boundaries.py','refine_block_separation.py'):
        path=Path(__file__).with_name(name);files[str(path)]=file_hash(path)
    snap=OUT/'target_source';snap.mkdir(exist_ok=True)
    for i,(path,sha) in enumerate(files.items()): shutil.copy2(path,snap/f'{i:02d}_{Path(path).name}')
    manifest=dict(training=list(train.values()),validation=val,reserved_test=test,pairs=old['pairs'],
        target='accepted gentle-v3 native720 frozen composite; training labels only',
        original_identity_loss=False,recipe_reproduction_error=error,recipe_bindings=files,
        validation_previously_reviewed=True,reserved_test_used_for_selection=False,
        inference='Only unchanged B1; no segmentation, quantizer, filter or high-res skip',
        contract_sha256=r.CONTRACT_SHA)
    write_json(OUT/'data.json',manifest)
    print('STYLE_DATA_READY',len(train),len(val),len(test),flush=True)


if __name__=='__main__':main()
