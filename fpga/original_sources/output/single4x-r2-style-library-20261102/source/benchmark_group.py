import time
import numpy as np
from PIL import Image
import library as l
from group_engine import GroupEngine
l.setup(123);entries=l.read_inventory()[:4];a=np.array(Image.open(l.OLD/'rule-preview/outdoor_source.png').convert('RGB'));x=l.tx(a[None])
for mode in ['new','old']:
    group=GroupEngine(folders=[l.D/'models'/e['id'] for e in entries]) if mode=='new' else GroupEngine(entries=entries)
    out=group(x);l.torch.cuda.synchronize();start=time.perf_counter()
    for _ in range(3):out=group(x)
    l.torch.cuda.synchronize();elapsed=(time.perf_counter()-start)/3
    for i,e in enumerate(entries):
        individual=l.PackedEngine(l.D/'models'/e['id']) if mode=='new' else l.Teacher(e)
        assert l.torch.equal(individual(x),out[i:i+1]),(mode,e['id'])
    print('FAST_GROUP_FULL720_EXACT',mode,'seconds_per_4styles_frame',round(elapsed,4),flush=True)
