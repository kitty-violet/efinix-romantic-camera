"""Verify saved inference pixels against checkpoints, including full 720p."""
from pathlib import Path
import ast,json
import numpy as np
from PIL import Image
from reconstruct import D,ROOT,Model,IntegerModel,torch,setup,tx,rgb,sha,save_json,costs

def main():
    setup(20261102);checks=[]
    for key in ['single4x-r2','original2x2x-r3']:
        folder=D/key;ck=torch.load(folder/'qat.pt',weights_only=True,map_location='cpu')
        model=Model(ck['single']).cuda();model.load_state_dict(ck['state_dict']);model.qat=True;model.eval();engine=IntegerModel(folder)
        with torch.inference_mode():
            gray=np.broadcast_to(np.arange(256,dtype=np.uint8)[:,None,None,None],(256,8,8,3)).copy()
            gx=tx(gray);a=engine(gx);b=model(gx);assert torch.equal(a,b)
            levels=(a[:,:,4,4].cpu().numpy()*128+128).astype(int)
            gray_error=int(np.abs(levels-np.arange(256)[:,None]).max())
            pixels=0
            for p in sorted((D/'review').glob('*_000_original.png')):
                img=np.asarray(Image.open(p).convert('RGB'));x=tx(img[None]);out=engine(x)
                assert torch.equal(model(x),out),(key,p)
                expected=np.asarray(Image.open(p.with_name(p.name.replace('_original.png','_'+key+'.png'))))
                assert np.array_equal(rgb(out),expected),(key,p)
                pixels+=expected.size
        assert costs(model.rows)['parameters']==sum(p.numel() for p in model.parameters())
        checks.append(dict(model=key,coefficient_sha256=sha(folder/'integer-coefficients.npz'),full720_checkpoint_vs_exported_vs_saved_scalar_comparisons=pixels,exact=True,constant_gray256_max_channel_error=gray_error,parameters=sum(p.numel() for p in model.parameters())))
    for p in (D/'source').glob('*.py'):ast.parse(p.read_text('utf-8'))
    report=json.loads((D/'verification.json').read_text('utf-8'))
    assert len(report['scenes'])==4 and all(v['video']['frames']==v['video']['unique_encoded_frames'] for s in report['scenes'] for v in s['models'].values())
    assert report['outputs_from_exported_integer_coefficients'] and not report['hardware_or_flash_modified']
    assert all(sha(p)==h for p,h in report['protected_pack_sha_unchanged'].items())
    save_json(D/'final-checks.json',dict(status='PASS_NUMERICAL_AND_MEDIA_CHECKS',checks=checks,source_static_check=True,no_protected_pack_changes=True,software_experiment_only=True))
    print('FINAL_CHECKS_PASS',json.dumps(checks,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
