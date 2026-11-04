"""Encode local soft-threshold smoothing inside existing DW/PW/ReLU layers.

For each latent colour lane: output = local_mean + ReLU(delta-T)
- ReLU(-delta-T). Uniform colours are unchanged; strong boundaries retain
their signed difference beyond T. No inference preprocessing is added.
"""
import json,argparse,copy
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import library as l
from refine import load_packed
from library import D,OLD,torch,save_json
from render_review import tiny_islands,rgb_batch,image_metrics,hash_array,FONT

@torch.no_grad()
def build(e,threshold=6):
    m=load_packed(D/'models'/e['id']);old=l.teacher_parameters(e['source'])
    def identity(n):
        m.w[n].zero_();m.b[n].zero_();getattr(m,n+'_m').fill_(2048)
        if m.w[n].shape[-1]==3:m.w[n][:,0,1,1]=1
        else:
            for c in range(min(m.w[n].shape[:2])):m.w[n][c,c,0,0]=1
    # Preserve the original colour feature encoding and exact constant gains.
    m.w['stem'].copy_(old.weight['stem']);m.b['stem'].copy_(old.bias['stem']);m.stem_m.copy_(old.stem_m.flatten())
    for name in ['down_dw','up1_low_dw','up1_high']:
        w=m.w[name];wi=l.HalfAway.apply(w*64).clamp(-127,127);wi[:,:,1,1]=0;centre=64-wi.sum((-1,-2))
        assert centre.abs().max()<=127
        wi[:,:,1,1]=centre;m.w[name].copy_(wi/64);getattr(m,name+'_m').fill_(2048);m.b[name].zero_()
    identity('up1_low_pw')
    # The stem/down spatial kernels remain the trained student coefficients.
    # Reuse its six positive/negative or split-intensity colour feature lanes.
    m.w['down_pw'].zero_();m.b['down_pw'].zero_();m.down_pw_m.fill_(2048)
    for c in range(24):m.w['down_pw'][c,c%6,0,0]=1
    k=torch.tensor([[0,.125,0],[.125,.5,.125],[0,.125,0]],device='cuda');eye=torch.zeros_like(k);eye[1,1]=1
    identity('res1_a_dw')
    for c in range(6):
        m.w['res1_a_dw'][6+c,0]=k
        m.w['res1_a_dw'][12+c,0]=eye-k
        m.w['res1_a_dw'][18+c,0]=k-eye
    identity('res1_a_pw')
    m.b['res1_a_pw'][12:24]=-threshold
    identity('res1_b_dw');m.w['res1_b_pw'].zero_();m.b['res1_b_pw'].zero_();m.res1_b_pw_m.fill_(2048)
    for c in range(6):
        m.w['res1_b_pw'][c,6+c,0,0]=1;m.w['res1_b_pw'][c,12+c,0,0]=1;m.w['res1_b_pw'][c,18+c,0,0]=-1
    for n in ['res2_a_dw','res2_a_pw','res2_b_dw','res2_b_pw']:identity(n)
    if e['id'].startswith('R8-'):
        m.w['res2_a_pw'].zero_();m.w['res2_a_pw'][:,:8].copy_(old.weight['down2__pw'][:,:8]);m.b['res2_a_pw'].copy_(old.bias['down2__pw']);m.res2_a_pw_m.copy_(old.down2__pw_m.flatten())
        m.w['res2_b_pw'].copy_(old.weight['res1_conv1__pw']);m.b['res2_b_pw'].copy_(old.bias['res1_conv1__pw']);m.res2_b_pw_m.copy_(old.res1_conv1__pw_m.flatten())
        # Once palette colours are chosen, preserve flat regions and crisp
        # boundaries instead of averaging unlike palette entries together.
        for n in ['up1_low_dw','up1_low_pw','up1_high']:identity(n)
    elif e['id']=='B03':
        m.w['res2_a_dw'].copy_(old.weight['res1_conv1__dw']);m.b['res2_a_dw'].copy_(old.bias['res1_conv1__dw']);m.res2_a_dw_m.copy_(old.res1_conv1__dw_m.flatten())
    m.w['rgb'].copy_(old.weight['rgb_head']);m.b['rgb'].copy_(old.bias['rgb_head']);m.rgb_m.copy_(old.rgb_head_m.flatten())
    m.qat=True;m.eval();del old
    return m

def main():
    p=argparse.ArgumentParser();p.add_argument('--ids',nargs='+',default=['PI02','R8-06']);p.add_argument('--threshold',type=int,default=6);a=p.parse_args();l.setup(20261102)
    sources=json.loads((OLD/'acceptance-four-scenes-5s-20261029/source_manifest.json').read_text('utf-8'))['scenes']
    for e in l.read_inventory():
        if e['id'] not in a.ids:continue
        m=build(e,a.threshold);folder=D/'clipped-merge-candidates'/str(a.threshold)/e['id'];l.export(m,folder);engine=l.PackedEngine(folder);current=l.PackedEngine(D/'models'/e['id']);teacher=l.Teacher(e)
        colors=np.asarray([[r,g,b] for r in [0,85,170,255] for g in [0,85,170,255] for b in [0,85,170,255]],np.uint8)
        flat=l.tx(np.broadcast_to(colors[:,None,None,:],(64,8,8,3)).copy())
        new_flat=rgb_batch(engine(flat));old_flat=rgb_batch(teacher(flat));flat_error=int(np.abs(new_flat.astype(int)-old_flat.astype(int)).max())
        records=[];canvas=Image.new('RGB',(1440,1200),'#111924');font=ImageFont.truetype(FONT,18)
        for row,s in enumerate(sources):
            n=len(s['pts'])//2;img=np.array(Image.open(OLD/'acceptance-four-scenes-5s-20261029/originals'/f"{s['scene']}_{n:03d}.png"));assert hash_array(img)==s['raw_input_frame_sha256'][n]
            x=l.tx(img[None]);old=rgb_batch(teacher(x))[0];base=rgb_batch(current(x))[0];new=rgb_batch(engine(x))[0]
            metrics={k:tiny_islands(z) for k,z in [('old',old),('base',base),('clipped',new)]};metrics['vs_old']=image_metrics(old,new);records.append(dict(scene=s['scene'],metrics=metrics))
            for col,(label,pic) in enumerate([('旧风格',old),('当前新结构',base),('局部弱边合并',new)]):
                ImageDraw.Draw(canvas).text((col*480+4,row*300+3),s['label']+' / '+label,font=font,fill='white');canvas.paste(Image.fromarray(pic).resize((480,270),Image.Resampling.BOX),(col*480,row*300+29))
        canvas.save(folder/'对比.jpg',quality=96);save_json(folder/'analysis.json',dict(style_id=e['id'],threshold=a.threshold,records=records,uniform_rgb64_max_channel_error_vs_old=flat_error,no_extra_operators=True,constructed_merger_with_trained_prefix=True))
        print('CLIPPED_MERGE_CHECK',e['id'],[(x['scene'],round(x['metrics']['base']['tiny_cells_fraction']*100,2),round(x['metrics']['clipped']['tiny_cells_fraction']*100,2)) for x in records],flush=True)
if __name__=='__main__':main()
