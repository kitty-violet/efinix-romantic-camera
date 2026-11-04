"""Check the executed joint model's data alignment and frozen/trainable paths."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from . import train_b1_minimum_area as m
from .train_b1_joint_field_colour import JointFieldColourTraining
from .runtime import configure_workspace, contract_guard, write_json
from .engine import set_reproducible


def main():
    configure_workspace(); contract_guard(); set_reproducible(20261006271)
    config=json.loads((m.OUT/'joint-field-colour-qat/config.json').read_text(encoding='utf-8'))
    args=SimpleNamespace(**config['args']);data=m.load_data(args.data_manifest)
    tr=JointFieldColourTraining(args,data)
    by={Path(r['path']).resolve():r for r in data['training']}
    for pair in data['pairs']:
        for key in ('current','next'):
            assert Path(pair[key+'_target']).resolve()==Path(by[Path(pair[key]).resolve()]['target']).resolve()
    records=[]
    for step,native in ((0,False),(1,False),(2,True)):
        tr.step=step;x,t,specs,flow,valid=tr.batch_style(native)
        for i,spec in enumerate(specs):
            row=tr.by_id.get(spec['id']) or tr.by_stem[spec['id']]
            xx,yy=spec['xy'];w,h=spec['wh']
            arrays=[m.t.r.read(row[key])[yy:yy+h,xx:xx+w] for key in ('path','target')]
            lab=m.load_region(row['region'])[yy:yy+h,xx:xx+w]
            if spec['flip']:arrays=[v[:,::-1] for v in arrays];lab=lab[:,::-1]
            for expected,actual in zip(arrays,(x[i],t[i])):
                reconstructed=(actual*128+128).round().byte().permute(1,2,0).cpu().numpy()
                assert np.array_equal(expected,reconstructed)
            assert np.array_equal(lab,tr.labels[i].cpu().numpy())
        records.append(dict(step=step,native720=native,temporal_pair=flow is not None,samples=specs,
                            source_target_region_alignment_error=0))
    # One real crop backward; no optimizer update or production checkpoint edit.
    tr.step=1;x,t,specs,_,_=tr.batch_style(False)
    p=tr.net(x)[0]
    loss,terms=tr.additional_loss(p[...,40:-40,40:-40],t[...,40:-40,40:-40],False)
    base,_=tr.appearance_loss(p,t,args.mode);assert float(base)==0
    loss.backward()
    gradients={key:float(w.grad.abs().sum()) for key,w in tr.net.weight.items() if w.requires_grad}
    assert gradients['up1_low_pw']>0 and gradients['up2_low_pw']>0
    fixed=['up1_high_dw','up2_low_dw','up2_high_dw','rgb_head']
    assert all(tr.net.weight[key].grad is None and not tr.net.weight[key].requires_grad for key in fixed)
    names=['research_b1_reconstruction.py','train_b1_target_latent.py','train_b1_count_regions.py',
           'train_b1_late_region_supervision.py','train_b1_minimum_area.py','train_b1_joint_region_colour.py',
           'train_b1_joint_field_colour.py','train_b1_aligned_decoder_regions.py','audit_b1_target_chain.py',
           'refine_b1_hierarchical_targets.py','add_b1_region_boundary_coverage.py','check_b1_training_execution.py',
           'probe_b1_joint_field_boundary.py','probe_b1_joint_decoder_latent.py','audit_b1_joint_constant_colours.py',
           'check_b1_joint_execution.py']
    for name in names:ast.parse(Path(__file__).with_name(name).read_text(encoding='utf-8-sig'))
    result=dict(status='PASS',batch_records=records,pair_target_paths_checked=2*len(data['pairs']),
        enabled_loss_terms=list(terms),reconstruction_appearance_loss=0,gradient_l1=gradients,
        frozen_layers=fixed,ast_files=names,contract_guard='PASS',optimizer_updates=0,
        scope='Alignment and executed code paths; no visual or temporal acceptance')
    write_json(m.OUT/'complete-chain-audit/joint_execution_checks.json',result)
    print('JOINT_EXECUTION_CHECKS_PASS',dict(pair_target_paths=result['pair_target_paths_checked'],
        source_target_region_alignment_error=0,ast_files=len(names),up1_gradient=gradients['up1_low_pw'],
        rgb_matrix_gradient=gradients['up2_low_pw']),flush=True)


if __name__=='__main__':main()
