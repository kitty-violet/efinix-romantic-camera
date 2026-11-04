"""Measure actual integer winner regions and compare their area distributions.

The labels are decoded before the normal interpolation, avoiding invented
fragmentation from AA colours. Areas scale by16 from180p to720p.
"""
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
from . import train_b1_minimum_area as m
from .minimum_area_blocks import components
from .runtime import configure_workspace,write_json,file_hash


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--device',choices=('cpu','cuda'),default='cuda');ap.add_argument('--ids',nargs='*');a=ap.parse_args()
    configure_workspace();torch.set_num_threads(2);cv2.setNumThreads(2)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True);assert state.get('sixteen_winner') or state.get('binary_regions') or state.get('ordered_regions') or state.get('manual_median_regions')
    net=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin').to(a.device)
    net.load_state_dict(state['model']);net.package['residual_encoding']=state['residual_encoding'];net.quantized=True
    actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(net.export()),device=a.device);records=[]
    rows=m.load_data()['validation']
    if a.ids:
        missing=set(a.ids)-{r['id'] for r in rows}
        assert not missing,f'Unknown validation IDs: {sorted(missing)}'
    for row in rows:
        if a.ids and row['id'] not in a.ids:continue
        rgb=m.t.r.read(row['path']);x=torch.tensor(rgb.transpose(2,0,1)[None].copy(),device=a.device,dtype=torch.float32)-128
        nodes,_=actual(x,True)
        if state.get('ordered_regions') or state.get('manual_median_regions'):
            node='up1_high_dw' if state.get('manual_median_regions') else 'up2_low_dw'
            masks=nodes[node][0,:15].cpu().numpy();assert np.all((masks==0)|(masks==127))
            ordered=[masks[start:end-1]>=masks[start+1:end] for start,end in ((0,7),(7,11),(11,15))]
            order_violation=float(np.concatenate(ordered,axis=0).size-np.concatenate(ordered,axis=0).sum())/sum(v.size for v in ordered)
            if not state.get('independent_region_boundaries'):assert order_violation==0
            labels=(masks[:7]>0).sum(0)*25+(masks[7:11]>0).sum(0)*5+(masks[11:15]>0).sum(0);factor=4
        elif state.get('binary_regions'):
            masks=nodes['up2_low_dw'][0,:4].cpu().numpy();assert np.all((masks==0)|(masks==127))
            labels=((masks>0)*np.array([1,2,4,8])[:,None,None]).sum(0);factor=4
        else:
            masks=nodes['up1_low_pw'][0].cpu().numpy();assert np.all(masks.sum(0)==127)
            labels=masks.argmax(0);factor=16
        regions=components(labels);areas=np.bincount(regions.ravel())*factor
        counts=np.bincount(labels.ravel(),minlength=200 if state.get('ordered_regions') or state.get('manual_median_regions') else 16)
        record=dict(id=row['id'],region_count=len(areas),min_area720=int(areas.min()),median_area720=float(np.median(areas)),
            pixel_fraction_below256=float(areas[areas<256].sum()/areas.sum()),region_fraction_below256=float((areas<256).mean()),
            actual_classes_used=int((counts>0).sum()),class_pixels_at_decision_grid=counts.tolist(),decision_grid_to720_area_factor=factor)
        if state.get('ordered_regions'):
            # A uniform wrong-colour output also scores well on purity, so keep
            # colour error alongside this structural diagnostic, never rank by
            # purity alone. Exclude a narrow boundary allowance, not whole areas.
            target_labels=m.load_region(row['region']).astype(np.int64)[::2,::2]
            boundary=np.zeros_like(target_labels,dtype=np.uint8)
            boundary[:,1:]|=(target_labels[:,1:]!=target_labels[:,:-1]).astype(np.uint8)
            boundary[1:]|=(target_labels[1:]!=target_labels[:-1]).astype(np.uint8)
            interior=cv2.dilate(boundary,np.ones((3,3),np.uint8))==0
            merged=target_labels[interior]*200+labels[interior]
            table=np.bincount(merged,minlength=(int(target_labels.max())+1)*200).reshape(-1,200)
            record['interior_target_region_nonmodal_colour_fraction']=float(1-table.max(1).sum()/max(1,table.sum()))
            record['interior_evaluated_pixel_fraction']=float(interior.mean())
            original=m.t.r.read(row['target']).astype(np.float32)
            output=nodes['rgb_head'][0].permute(1,2,0).cpu().numpy()+128
            record['original_target_rgb_mae']=float(np.abs(output-original).mean())
        if state.get('ordered_regions'):record['independent_bits_order_violation_fraction']=order_violation
        records.append(record);print('SIXTEEN_REGION_AUDIT',{k:v for k,v in record.items() if k!='class_pixels_at_decision_grid'},flush=True)
    write_json(a.output,dict(checkpoint=a.checkpoint,sha256=file_hash(a.checkpoint),records=records,
        mean_small_region_pixel_fraction=float(np.mean([r['pixel_fraction_below256'] for r in records])),
        decoded_from_actual_integer_nodes=True,area_is_decision_grid_scaled_to720=True,
        global_minimum_area_guaranteed=False,visual_acceptance=False,device=a.device,selected_ids=a.ids))


if __name__=='__main__':main()
