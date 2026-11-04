"""Audit actual adjacent targets under the same valid optical-flow mask.

Do not interpret a teacher's own inconsistency as a permitted output flicker.
This measures the old allowance; it does not make an acceptance claim.
"""
import json
import numpy as np,torch
from .train_luma_regions import ROOT,RUN,labels,centers,reconstruct,warp
from .train_b1_distillation import as_tensor,read_image,read_flow
from .runtime import file_hash,write_json


@torch.no_grad()
def main():
    torch.set_num_threads(2);root=ROOT/'runs'/RUN;out=root/'region-temporal-target-audit-v1';out.mkdir(exist_ok=False)
    data=json.loads((root/'native-mid-fp32/data.json').read_text(encoding='utf-8'));rows=[]
    for i,r in enumerate(data['pairs']):
        x=torch.stack([as_tensor(read_image(r[k])) for k in ['current','next']]);t=torch.stack([as_tensor(read_image(r[k])) for k in ['current_target','next_target']]);f,m=read_flow(r['flow'])
        flow=torch.from_numpy(f).permute(2,0,1)[None];mask=torch.from_numpy(m)[None,None];xd=(x[1:]-warp(x[:1],flow)).abs().mean(1,keepdim=True)*128
        valid=(mask>.5)&(xd<8);valid[...,:48,:]=False;valid[...,-48:,:]=False;valid[...,:,:48]=False;valid[...,:,-48:]=False
        lab=labels(t);q=reconstruct(*[c[l] for c,l in zip(centers(t),lab)])
        td=(t[1:]-warp(t[:1],flow)).abs().mean(1,keepdim=True)*128;qd=(q[1:]-warp(q[:1],flow)).abs().mean(1,keepdim=True)*128
        def stats(z):
            a=z[valid].numpy();return dict(mean=float(a.mean()),p95=float(np.percentile(a,95)),over8_fraction=float((a>8).mean()))
        rows.append(dict(index=i,source=r['source'],input=stats(xd),teacher=stats(td),quantized_teacher=stats(qd),teacher_excess=stats((qd-xd).clamp_min(0))))
        if (i+1)%20==0:print('TEMPORAL_TARGET_AUDIT',i+1,flush=True)
    summary={k:{s:float(np.mean([r[k][s] for r in rows])) for s in rows[0][k]} for k in ['input','teacher','quantized_teacher','teacher_excess']}
    write_json(out/'report.json',dict(summary=summary,records=rows,flow_is_estimate=True,mask='original valid flow + input mean RGB difference <8;48px border excluded',script_sha256=file_hash(__file__),data_sha256=file_hash(root/'native-mid-fp32/data.json')));print(summary,flush=True)


if __name__=='__main__':main()
