"""Same TRAIN crops/objective, identity versus damped covariance coordinates."""
import argparse,copy,json,time,shutil,math
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .train_b1_trainable_luma_readout import ReadoutB1
from .train_b1_rendered_luma_readout import RenderedLumaReadoutTraining
from .train_b1_relaxed_count_interfaces import RelaxedCountInterfaceTraining
from .train_b1_spatial_rgb_readout import SpatialRGBReadoutTraining
from .train_b1_balanced_inverse_geometry import BalancedInverseGeometryTraining
from .train_b1_full_rgb_geometry import FullRGBGeometryTraining
from .train_b1_complete_rgb_geometry import CompleteRGBGeometryTraining
from .b1_refine_live_luma import refine as refine_live_luma
from .b1_fine_spatial_decoder import install as install_fine_decoder
from .b1_spare_spatial_bank import install as install_spares,compensate_head,masks as spare_masks
from .fit_b1_current_rim_priority import optimizer as old_optimizer
from .centered_b1_optimizer import calibration_means
from .whitened_b1_optimizer import WhitenedB1Adam,patch_statistics,whitener,regression
from .b1_joint_interface_targets import joint_interface_fields
from .runtime import write_json,file_hash
from .runtime import configure_workspace
from .engine import set_reproducible
from . import train_b1_minimum_area as m
from .train_b1_late_region_supervision import executed_sources
import types


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--mining',required=True);ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=1000);ap.add_argument('--preflight-only',action='store_true');ap.add_argument('--resume-from',default='');ap.add_argument('--interval',type=int,default=100);ap.add_argument('--objective',choices=['rendered','relaxed-count','spatial-rgb','balanced-inverse','full-rgb','complete-rgb'],default='rendered');ap.add_argument('--only-whitened',action='store_true');ap.add_argument('--spare-bank',action='store_true');ap.add_argument('--train-stride',action='store_true');ap.add_argument('--refined-targets',default='');ap.add_argument('--free-decoder',action='store_true');ap.add_argument('--refine-live-luma',action='store_true');ap.add_argument('--cosine-decay',action='store_true');a=ap.parse_args()
    training_class={'rendered':RenderedLumaReadoutTraining,'relaxed-count':RelaxedCountInterfaceTraining,'spatial-rgb':SpatialRGBReadoutTraining,'balanced-inverse':BalancedInverseGeometryTraining,'full-rgb':FullRGBGeometryTraining,'complete-rgb':CompleteRGBGeometryTraining}[a.objective]
    if a.objective=='balanced-inverse':assert a.refined_targets,'Inverse-code balance must use verified refined labels'
    configure_workspace();set_reproducible(2026100722);torch.set_num_threads(2);regression()
    state,initial=load(a.checkpoint);initial.__class__=ReadoutB1;initial.cuda();out=Path(a.out);out.mkdir(exist_ok=False)
    data=m.load_data(a.data);byid={r['id']:r for r in data['training']};mined=json.loads(Path(a.mining).read_text())['records'];specs=[mined[i] for i in (1,5,6)]
    def inputs(row,sl):
        def tensor(path):return torch.from_numpy(m.t.r.read(path)[sl].transpose(2,0,1).copy()).float().cuda()/128-1
        with np.load(row['native_luma_codes']) as z:code=torch.from_numpy(z['codes'][sl].astype(np.int64)).cuda()
        return tensor(row['path']),tensor(row['target']),torch.from_numpy(m.load_region(row['region'])[sl].astype(np.int64)).cuda(),code
    batches=[]
    for spec in specs:
        row=byid[spec['row']['id']];x0,y0=spec['xy'];w,h=spec['wh'];batches.append(inputs(row,np.s_[y0:y0+h,x0:x0+w]))
    x,t,labels,code=[torch.stack(z) for z in zip(*batches)];truth=joint_interface_fields(code)
    if a.refined_targets:
        refined=torch.load(a.refined_targets,map_location='cuda',weights_only=True)
        assert refined['training_only'] and refined['not_source_prediction'] and refined['specs']==specs
        assert refined['codes'].shape==truth[:,:14].shape
        truth=refined['codes'].float()
    if a.spare_bank:
        assert a.objective=='spatial-rgb' and not a.resume_from,'First bank probe is a fresh optimizer warm start'
        with torch.no_grad():
            old_nodes=initial(x,all_nodes=True);install_spares(initial);new_nodes=initial(x,all_nodes=True)
            assert torch.equal(old_nodes['rgb_head'],new_nodes['rgb_head'])
            assert all(torch.equal(old_nodes[k],v) for k,v in new_nodes.items() if k not in ('up2_low_pw','up2_nn','up2_high_dw'))
        print('SPARE_BANK_INITIAL_RGB_EXACT',flush=True)
    if a.free_decoder:
        assert a.objective in ('full-rgb','complete-rgb') and not a.spare_bank and not a.refined_targets and not a.resume_from
        with torch.no_grad():
            old_nodes=initial(x,all_nodes=True)
            if not state.get('fine_full_decoder_trainable'):install_fine_decoder(initial)
            new_nodes=initial(x,all_nodes=True)
            assert torch.equal(old_nodes['rgb_head'],new_nodes['rgb_head'])
            assert all(torch.equal(old_nodes[k],v) for k,v in new_nodes.items() if k not in ('up2_low_pw','up2_nn','up2_high_dw'))
        print('FINE_DECODER_INITIAL_RGB_AND_EXISTING_NODES_EXACT',flush=True)
    if a.refine_live_luma:
        assert a.free_decoder and not a.resume_from
        with torch.no_grad():
            old_nodes=initial(x,all_nodes=True);info=refine_live_luma(initial);new_nodes=initial(x,all_nodes=True)
            assert all(torch.equal(old_nodes[k],v) for k,v in new_nodes.items())
        print('LIVE_LUMA_PRECISION_ALL_NODES_EXACT',info,flush=True)
    full=tuple(v[None] for v in inputs(byid[specs[0]['row']['id']],np.s_[:,:]));fulltruth=joint_interface_fields(full[3])
    means=calibration_means(initial,x);old=old_optimizer(initial,means);groups=[dict(params=list(g['params']),lr=g['lr'],**({'row_gain':g['row_gain']} if 'row_gain' in g else {})) for g in old.param_groups]
    if a.train_stride:
        for k in ('down1__dw','down2__dw'):initial.weight[k].requires_grad_(True);initial.bias[k].requires_grad_(True)
    if a.free_decoder:
        for k,w in initial.weight.items():
            if k.startswith(('up1_high','up2_','rgb_head')):w.requires_grad_(True);initial.bias[k].requires_grad_(True)
    before=initial.export();stats=patch_statistics(initial,x,4096);assert initial.export()==before
    covreport={}
    for k,s in stats.items():
        _,r=whitener(s['covariance']);covreport[k]={name:v.tolist() if torch.is_tensor(v) else v for name,v in r.items()}
    paths,_=executed_sources(training_class);paths=sorted(set(paths+[Path(__file__).resolve()]));frozen={str(p):file_hash(p) for p in paths}
    for p in paths:shutil.copy2(p,out/p.name)
    resumed=torch.load(a.resume_from,map_location='cuda',weights_only=True) if a.resume_from else None
    if resumed:assert resumed.get('whitened_update') and resumed.get('local_fit_only') and resumed['step']<a.steps
    if resumed:assert resumed.get('local_objective','rendered')==a.objective,'Objective changed: use fresh optimizer warm start'
    write_json(out/'config.json',dict(args=vars(a),source_sha256=file_hash(__file__),optimizer_sha256=file_hash(Path(__file__).with_name('whitened_b1_optimizer.py')),train_specs=specs,covariance=covreport,damping=.05,RMS_transform_gain=1,no_extra_inference=True,code=frozen,resume_sha256=file_hash(a.resume_from) if a.resume_from else None,refined_targets_sha256=file_hash(a.refined_targets) if a.refined_targets else None,native_preflight_geometry='analytic labels; local training uses supplied refined labels' if a.refined_targets else 'analytic labels'))
    def opt_for(net,enabled):
        gg=[]
        for k,w in net.weight.items():
            extra=a.spare_bank and k in ('up2_low_pw','rgb_head')
            live=initial.weight[k].requires_grad or extra;w.requires_grad_(live);net.bias[k].requires_grad_(live and not(a.spare_bank and k=='rgb_head'))
            if live:
                s=(getattr(net,k+'_m')/131072).float().flatten()
                gg.append(dict(params=[w],lr=.0001,row_gain=(s*256)[:,None,None,None],unwhitened=extra))
                if net.bias[k].requires_grad:gg.append(dict(params=[net.bias[k]],lr=.008))
                if extra:
                    mask=spare_masks(net)[k];w.register_hook(lambda g,mask=mask:g*mask)
                    if k=='up2_low_pw':
                        bm=mask.flatten(1).amax(1);net.bias[k].register_hook(lambda g,bm=bm:g*bm)
        return WhitenedB1Adam(gg,net,stats,enabled)
    def objective(net,batch,tt,native):
        xx,tar,lab,codes=batch;nodes=net(xx,all_nodes=True);obj=object.__new__(training_class)
        obj.net=net;obj.nodes=nodes;obj.labels=lab;obj.class_labels=codes;obj.distance_cache=lambda c,g:tt
        obj.motion_fraction=xx.new_zeros(());obj.args=types.SimpleNamespace(classification=2.)
        cut=(lambda z:z) if native else (lambda z:z[...,40:-40,40:-40]);p=nodes['rgb_head']/128
        loss,v=obj.additional_loss(cut(p),cut(tar),native)
        return loss,v,p,nodes
    def update(net,opt,batch,tt,native=False):
        opt.zero_grad();loss,v,p,nodes=objective(net,batch,tt,native);loss.backward();opt.prepare_gradients()
        grad=torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],5,error_if_nonfinite=True);opt.step()
        if a.spare_bank:
            with torch.no_grad():
                k='up2_low_pw';s=(getattr(net,k+'_m')/131072).float().flatten()[:,None,None,None]
                net.weight[k][3:].copy_(net.weight[k][3:].maximum(-127*s[3:]).minimum(127*s[3:]))
                compensate_head(net,initial.bias['rgb_head'])
                assert torch.equal(net.weight[k][:3],initial.weight[k][:3]) and torch.equal(net.weight['rgb_head'][:,:3],initial.weight['rgb_head'][:,:3])
        return float(loss.detach()),float(grad)
    batch=(x,t,labels,code)
    # Actual GPU full-frame and crop updates, fresh-instance next-update resume.
    net=copy.deepcopy(initial);opt=opt_for(net,True)
    if resumed:net.load_state_dict(resumed['model']);opt.load_state_dict(resumed['optimizer'])
    native=update(net,opt,full,fulltruth,True);crop=update(net,opt,batch,truth)
    saved=dict(model=copy.deepcopy(net.state_dict()),optimizer=copy.deepcopy(opt.state_dict()));torch.save(saved,out/'preflight.pt')
    expected=update(net,opt,batch,truth);other=copy.deepcopy(initial);op=opt_for(other,True)
    disk=torch.load(out/'preflight.pt',map_location='cuda',weights_only=True)
    other.load_state_dict(disk['model']);op.load_state_dict(disk['optimizer']);got=update(other,op,batch,truth)
    errors={k:float((net.state_dict()[k]-v).abs().max()) for k,v in other.state_dict().items()}
    err=max(errors.values());write_json(out/'resume_diagnostic.json',dict(error=err,expected=expected,actual=got,nonzero={k:v for k,v in errors.items() if v}))
    print('WHITENED_RESUME',err,expected,got,flush=True);assert err==0 and expected==got
    with torch.no_grad():
        nodes=net(x,all_nodes=True);rt=m.t.r.IntegerB1(m.t.r.runtime_from_image(net.export()),device='cuda');actual,_=rt(net.round(x*128),True);assert all(torch.equal(nodes[k],v) for k,v in actual.items())
    write_json(out/'preflight.json',dict(status='PASS',native=native,crop=crop,resume_error=err,actual_disk_checkpoint_loaded=True,actual_all_nodes_error=0,bytes=len(net.export()),precision='FP32_noAMP_noTF32'))
    print('WHITENED_GPU_PREFLIGHT_PASS native/crop/next_update_resume0/actual_nodes0',flush=True)
    del net,opt,other,op,rt,nodes,actual;torch.cuda.empty_cache()
    if a.preflight_only:return
    images=[x,t];records=[];started=time.monotonic()
    image_labels=['TRAIN原图','目标']
    for enabled in ((True,) if resumed or a.only_whitened else (False,True)):
        net=copy.deepcopy(initial);opt=opt_for(net,enabled);history=[];base_lrs=[g['lr'] for g in opt.param_groups]
        if resumed:net.load_state_dict(resumed['model']);opt.load_state_dict(resumed['optimizer'])
        start=int(resumed['step']) if resumed else 0
        for step in range(start,a.steps+1):
            if a.cosine_decay:
                factor=.05+.95*(1+math.cos(math.pi*step/a.steps))/2
                for group,base_lr in zip(opt.param_groups,base_lrs):group['lr']=base_lr*factor
            if step%a.interval==0 or step==a.steps:
                assert all(file_hash(p)==v for p,v in frozen.items()),'Executing source changed'
                with torch.no_grad():
                    loss,v,p,nodes=objective(net,batch,truth,False);r=dict(step=step,loss=float(loss),mae=float((p[...,40:-40,40:-40]-t[...,40:-40,40:-40]).abs().mean()*128),rim=float(v['two_pixel_region_centred_bright_shoulder']),terms={k:float(z) for k,z in v.items()})
                history.append(r);write_json(out/f'white{int(enabled)}_history.json',history);print('WHITENED_FIT',enabled,r,flush=True)
                board=Image.new('RGB',(320*(len(images)+1),350*len(x)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
                arm_label=('全RGB几何·实际B1' if a.free_decoder else '逆解标签训练·实际源预测' if a.refined_targets else ('采样系数可学习' if a.train_stride else '五个末端通道启用' if a.spare_bank else '协方差坐标' if enabled else '均值中心化')+' / '+{'relaxed-count':'联合档位','spatial-rgb':'RGB边界','rendered':'原目标','full-rgb':'最终RGB'}[a.objective])
                for i in range(len(x)):
                    for j,z in enumerate(images+[p]):board.paste(m.t.r.c.rgb_image(z[i]),(320*j,350*i+30));draw.text((320*j+2,350*i+3),(image_labels+[arm_label])[j],font=font,fill='white')
                board.save(out/f'white{int(enabled)}_{step:06d}.png')
                saved=copy.deepcopy(state);saved.update(model=net.state_dict(),optimizer=opt.state_dict(),step=step,local_fit_only=True,whitened_update=enabled,local_objective=a.objective,spare_spatial_bank=a.spare_bank,trainable_stride_coefficients=a.train_stride,refined_geometry_targets=bool(a.refined_targets),fine_full_decoder_trainable=a.free_decoder,visual_acceptance=False,live_luma_precision_refined=a.refine_live_luma,cosine_decay=a.cosine_decay)
                torch.save(saved,out/f'white{int(enabled)}_{step:06d}.pt')
            if step==a.steps:break
            update(net,opt,batch,truth)
        with torch.no_grad():
            rt=m.t.r.IntegerB1(m.t.r.runtime_from_image(net.export()),device='cuda');actual,_=rt(net.round(x*128),True);assert all(torch.equal(nodes[k],v) for k,v in actual.items())
        saved=copy.deepcopy(state);saved.update(model=net.state_dict(),optimizer=opt.state_dict(),step=step,local_fit_only=True,whitened_update=enabled,local_objective=a.objective,spare_spatial_bank=a.spare_bank,trainable_stride_coefficients=a.train_stride,refined_geometry_targets=bool(a.refined_targets),fine_full_decoder_trainable=a.free_decoder,visual_acceptance=False)
        torch.save(saved,out/f'white{int(enabled)}.pt');images.append(p.detach());image_labels.append(arm_label);records.append(dict(whitening=enabled,history=history,actual_all_nodes_error=0))
        write_json(out/'review.json',dict(records=records,local_fit_not_generalization=True,visual_acceptance=False,seconds=time.monotonic()-started))
    print('WHITENED_FIT_COMPLETE',flush=True)


if __name__=='__main__':main()
