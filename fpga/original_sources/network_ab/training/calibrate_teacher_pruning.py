"""Input-feature least-squares compensation for physically pruned channels."""
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw
import torch
from torch import nn

from .prune_animation_teacher import make_indices,build_pruned,profile,TEACHER_SHA
from .trainable_animegan import TrainableAnimeGAN
from .train_b1_distillation import as_tensor,to_rgb,read_image
from .runtime import configure_workspace,file_hash,write_json

RUN='teacher-pruning-calibrated-20261003'


def main():
    root=configure_workspace();run=root/'runs'/RUN;run.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    teacher_path=root/'runs/texture-recovery-20261003/final/refined_style_fp32.pt';assert file_hash(teacher_path)==TEACHER_SHA
    state=torch.load(teacher_path,map_location='cpu',weights_only=True)['generator']
    teacher=TrainableAnimeGAN(root).cuda().eval();teacher.load_state_dict(state)
    manifest=json.loads((root/'runs/teacher-pruning-20261003/data_manifest.json').read_text(encoding='utf-8'))
    write_json(run/'data_manifest.json',manifest)
    groups={}
    for r in manifest['pairs']:groups.setdefault(r['source'],[]).append(r)
    chosen=[rows[len(rows)//2] for _,rows in sorted(groups.items())]
    covariance={};counts={}
    def capture(name):
        def hook(module,args):
            a=args[0].detach()[0].flatten(1)
            # Deterministic uniform spatial sample, not validation content.
            n=min(a.shape[1],8192);index=torch.linspace(0,a.shape[1]-1,n,device=a.device).long()
            a=a[:,index].double();xx=(a@a.T).cpu()
            covariance[name]=covariance.get(name,torch.zeros_like(xx))+xx
            counts[name]=counts.get(name,0)+n
        return hook
    hooks=[m.register_forward_pre_hook(capture(n)) for n,m in teacher.named_modules() if isinstance(m,nn.Conv2d)]
    with torch.inference_mode():
        for row in chosen:
            x=as_tensor(read_image(row['current']))[None].cuda();teacher(x)
            print('CALIBRATION_SOURCE',Path(row['source']).name,flush=True)
    for h in hooks:h.remove()
    torch.save(dict(covariance=covariance,counts=counts),run/'input_second_moments.pt')
    write_json(run/'calibration_sources.json',dict(rows=chosen,source_isolation=True,method='7 training frames; uncentered feature second moments; 8192 uniform spatial samples per convolution per frame.'))
    records=[]
    for keep in (.75,.5,.25):
        indices,_=make_indices(teacher,keep);net=build_pruned(root,indices,state);layers=[]
        for name,m in teacher.named_modules():
            if not isinstance(m,nn.Conv2d):continue
            cin,cout=m.in_channels,m.out_channels
            ii=torch.tensor(indices.get(str(cin),list(range(cin))),dtype=torch.long)
            oo=torch.tensor(indices.get(str(cout),list(range(cout))),dtype=torch.long)
            xx=covariance[name]/counts[name]
            cov_kk=xx[ii][:,ii];ridge=float(cov_kk.trace()/len(ii))*1e-4+1e-12
            # Approximate all original input channels using the retained subset.
            A=torch.linalg.solve(cov_kk+ridge*torch.eye(len(ii),dtype=torch.float64),xx[ii]).T
            A[ii]=torch.eye(len(ii),dtype=torch.float64)
            w=m.weight.detach().cpu()[oo].double()
            compensated=torch.einsum('oihw,ik->okhw',w,A).float()
            new=net.get_submodule(name)
            with torch.no_grad():new.weight.copy_(compensated)
            layers.append(dict(name=name,retained_input=len(ii),old_input=cin,ridge=ridge,
                map_max=float(A.abs().max()),input_reconstruction_error=float((xx-A@xx[ii]-xx[:,ii]@A.T+A@xx[ii][:,ii]@A.T).trace()/xx.trace().clamp_min(1e-12))))
        net=net.cuda().eval();name=f'keep{round(keep*100):02d}';out=run/name;out.mkdir()
        p=out/'initial_fp32.pt'
        torch.save(dict(schema='calibrated-structured-pruned-teacher-fp32-v1',indices=indices,
            generator={k:v.cpu() for k,v in net.state_dict().items()},source_teacher_sha256=TEACHER_SHA,
            keep_channel_fraction=keep,calibration=True,recovery_steps=0,production=False),p)
        info=profile(net,x)
        with torch.inference_mode():
            for row in manifest['validation']:
                source=read_image(row['source']);target=read_image(row['target'])
                prediction=to_rgb(net(as_tensor(source)[None].cuda())[0])
                Image.fromarray(prediction).save(out/(row['id']+'.png'))
                if row['id'] not in ('val_stairs_090','val_plaza_090','val_new_hallway_120'):continue
                xx,yy=(424,320) if 'stairs' in row['id'] else (80,400) if 'plaza' in row['id'] else (560,150)
                panel=Image.new('RGB',(960,348),'#19202a');draw=ImageDraw.Draw(panel)
                for j,(v,label) in enumerate([(source,'Source'),(target,'Teacher target'),(prediction,name+' calibrated / no recovery')]):
                    panel.paste(Image.fromarray(v[yy:yy+320,xx:xx+320]),(j*320,28));draw.text((j*320+8,8),label,fill='white')
                panel.save(out/(row['id']+'_comparison.png'))
        records.append(dict(name=name,keep=keep,indices=indices,layers=layers,profile=info,weights_sha256=file_hash(p)))
        write_json(run/'calibrated_pruning.json',dict(status='COMPLETE' if len(records)==3 else 'CALIBRATING',records=records))
        print('CALIBRATED_PRUNING',name,flush=True)
        del net;torch.cuda.empty_cache()
    write_json(run/'source_binding.json',dict(code={str(Path(__file__)):file_hash(__file__)},teacher_sha256=TEACHER_SHA,
        note='No raw/target file modified, no output postprocessing. Calibration derives new smaller convolution weights, not a sparse mask.'))


if __name__=='__main__':main()
