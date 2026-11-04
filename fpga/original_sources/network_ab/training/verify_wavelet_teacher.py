"""GPU comparison with pinned upstream full inference, not just local math."""
import importlib.util
import json
from pathlib import Path
import sys
import types

from PIL import Image
import torch

from .runtime import configure_workspace, file_hash, write_json
from .wavelet_game_teacher import WaveletTeacher
from .probe_wavelet_game import tensor, verify_math


def main():
    root=configure_workspace();run=root/'runs/wavelet-game-probe-20261003'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    net=WaveletTeacher(root)
    sys.path.insert(0,str(net.directory))
    # Pin the package location so an unrelated package named utils cannot win.
    utils=types.ModuleType('utils');utils.__path__=[str(net.directory/'utils')];sys.modules['utils']=utils
    spec=importlib.util.spec_from_file_location('wct2_official_transfer',net.directory/'transfer.py')
    upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
    content=tensor(Image.open(run/'plaza_stone/source.png').resize((320,176),Image.Resampling.BICUBIC))
    style=tensor(Image.open(run/'references/garden_720.png').resize((320,176),Image.Resampling.BICUBIC))
    results=[]
    for option in ('cat5','sum'):
        if option!='cat5':net=WaveletTeacher(root,option=option)
        author=upstream.WCT2.__new__(upstream.WCT2)
        author.encoder,author.decoder=net.encoder,net.decoder
        author.device=torch.device('cuda');author.verbose=False;author.transfer_at={'encoder'}
        with torch.no_grad():
            net.prepare_style('test',style)
            actual=net(content,'test',('encoder',),.7)
            import numpy as np
            expected=author.transfer(content,style,np.asarray([]),np.asarray([]),alpha=.7).clamp(0,1)
            maximum=float((actual-expected).abs().max());mean=float((actual-expected).abs().mean())
            if not torch.allclose(actual,expected,atol=2e-4,rtol=2e-4):raise AssertionError((option,maximum,mean))
            repeated=net(content,'test',('encoder',),.7)
            if not torch.equal(actual,repeated):raise AssertionError('Same-frame inference not deterministic')
        results.append({'option':option,'full_pipeline_max_abs':maximum,'mean_abs':mean,'repeat_bit_exact':True})
        print(results[-1],flush=True)
    write_json(run/'verification.json',{'status':'PASS','upstream_math':verify_math(net),'full_pipeline':results,
        'source_sha256':file_hash(__file__),'wrapper_sha256':file_hash(Path(__file__).with_name('wavelet_game_teacher.py'))})


if __name__=='__main__':main()
