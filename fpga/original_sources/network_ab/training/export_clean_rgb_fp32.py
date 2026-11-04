"""Export genuine generator weights, explicitly not a board-ready package."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from .clean_rgb_skip_net import CleanRGBNet,plan,budget
from .train_b1_distillation import read_image,as_tensor,to_rgb
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True);a=ap.parse_args();configure_workspace();contract_guard();set_reproducible(2026100465)
    s=torch.load(a.checkpoint,map_location='cpu',weights_only=True);schema=s['config']['graph_schema']
    assert schema in ('direct-teacher-rgb-skip-v1','contour-region-bottleneck-v1') and s['scales'] is None and all(file_hash(k)==v for k,v in s['binding']['code'].items())
    net_class,graph_function,budget_function=CleanRGBNet,plan,budget
    if schema=='contour-region-bottleneck-v1':
        from .contour_region_net import ContourRegionNet,plan as contour_plan,budget as contour_budget
        net_class,graph_function,budget_function=ContourRegionNet,contour_plan,contour_budget
    rows=[json.loads(line) for line in (a.checkpoint.parent/'fp32_metrics.jsonl').read_text().splitlines()];assert [r['step'] for r in rows]==list(range(1,s['step']+1))
    out=a.checkpoint.parent/f'export-{s["step"]:06d}';out.mkdir(exist_ok=False);path=out/'generator_fp32.pt'
    payload=dict(model=s['model'],step=s['step'],graph=graph_function(),binding=s['binding'],source_checkpoint=str(a.checkpoint),source_sha256=file_hash(a.checkpoint),inference=net_class.__name__+'; RGB uint8 normalized (x-128)/128; NCHW; native1280x720; clamp signed output and add128',palette=False,postfilter=False,temporal_state=False,critic_deployed=False,teacher_deployed=False,style_accepted=False,fpga_deployable_verified=False)
    torch.save(payload,path);n=net_class().cuda().eval();n.load_state_dict(s['model']);reload=net_class().cuda().eval();reload.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model']);data=json.loads((a.checkpoint.parent/'data.json').read_text(encoding='utf-8'));checks=[]
    for r in data['validation']:
        x=as_tensor(read_image(r['source']))[None].cuda();y=n(x);z=reload(x);error=float((y-z).abs().max());assert error==0
        expected=read_image(a.checkpoint.parent/f'fp32_{s["step"]:06d}'/(r['id']+'.png'));pixel=int(np.abs(to_rgb(z[0]).astype(np.int16)-expected.astype(np.int16)).max());assert pixel==0
        checks.append(dict(id=r['id'],shape=list(z.shape),reload_max_error=error,png_max_error=pixel))
    write_json(out/'verification.json',dict(status='FP32_EXPORT_RELOAD_PASS_NOT_STYLE_OR_BOARD_ACCEPTANCE',weights_sha256=file_hash(path),file_bytes=path.stat().st_size,budget=budget_function(),checks=checks,steps=s['step'],native_updates=sum(r['native'] for r in rows),actual_optimizer_seconds=sum(r['seconds'] for r in rows),QAT=False,RTL=False,fit=False,timing=False))
    write_json(out/'graph.json',graph_function());print('DIRECT_RGB_EXPORTED',file_hash(path),flush=True)


if __name__=='__main__':main()
