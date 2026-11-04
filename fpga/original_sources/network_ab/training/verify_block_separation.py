"""Independent process replay from real neural prediction through frozen code."""
import importlib,json,sys,types,time
from pathlib import Path
import cv2,numpy as np
from PIL import Image
from .runtime import configure_workspace,file_hash,write_json
from .engine import set_reproducible

OUT=Path(r'E:\fpga\model-training-workspace\runs\region-detail-deploy-20261004\block-separation-20261004-v3')


def main():
    configure_workspace();set_reproducible(2026100492);cv2.setNumThreads(2)
    config=json.loads((OUT/'experiment.json').read_text(encoding='utf-8'))
    for name,sha in config['code'].items():assert file_hash(OUT/'source_code'/name)==sha
    package=types.ModuleType('frozen_block_run');package.__path__=[str(OUT/'source_code')];sys.modules[package.__name__]=package
    direction=importlib.import_module('frozen_block_run.directional_paint')
    block=importlib.import_module('frozen_block_run.block_separation')
    paint=importlib.import_module('frozen_block_run.painting_parameters')
    adapter=np.load(OUT/'color_adapter.npz')['lab_chroma_affine'];old=Path(config['previous_package']);rows=[]
    for case in config['inputs']:
        key=case['id'];source=old/(key+'_student.png');assert file_hash(source)==case['neural_prediction']['input_sha256']
        assert file_hash(OUT/(key+'_surface.png'))==case['surface_sha256']
        rgb=np.array(Image.open(source).convert('RGB'))
        base=direction.surface(paint.tone(rgb,adapter),direction.PRESETS['direction4'])
        saved=np.array(Image.open(OUT/(key+'_surface.png')))
        err=int(np.abs(base.astype(int)-saved.astype(int)).max());assert err==0
        result=block.transform_surface(base,'balanced');published=np.array(Image.open(OUT/(key+'_balanced.png')))
        delta=int(np.abs(result.astype(int)-published.astype(int)).max());assert delta==0
        rows.append(dict(id=key,recomputed_surface_max_error=err,final_rgb_max_error=delta))
        print('BLOCK_REPLAY',key,err,delta,flush=True)
    validation=json.loads((OUT/'validation.json').read_text(encoding='utf-8'))
    validation['independent_replay']=dict(frozen_code_hashes_pass=True,replay_source='cached real learned RGB predictions; tone+CUDA surface+block transform recomputed',
                                          neural_forward_repeated=False,precision='FP32, AMP disabled, TF32 disabled',records=rows)
    write_json(OUT/'validation.json',validation)
    print('BLOCK_REPLAY_DONE',flush=True)


if __name__=='__main__':main()
