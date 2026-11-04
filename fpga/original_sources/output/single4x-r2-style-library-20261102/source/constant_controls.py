"""Palette and neutral-gray diagnostics on actual exported BINs."""
import numpy as np
import library as l
from render_review import rgb_batch

def main():
    l.setup(20261102);gray=np.repeat(np.arange(256,dtype=np.uint8)[:,None],3,1)
    colours=np.asarray([[r,g,b] for r in [0,85,170,255] for g in [0,85,170,255] for b in [0,85,170,255]],np.uint8)
    samples=np.r_[gray,colours];patches=l.tx(np.broadcast_to(samples[:,None,None,:],(320,8,8,3)).copy());rows=[]
    for e in l.read_inventory():
        if e['id']=='BW_LINE':continue
        engine=l.PackedEngine(l.D/'models'/e['id']);teacher=l.Teacher(e)
        new=rgb_batch(engine(patches))[:,4,4];old=rgb_batch(teacher(patches))[:,4,4]
        row=dict(id=e['id'],gray256_max_channel_error_vs_old=int(np.abs(new[:256].astype(int)-old[:256]).max()),gray256_mean_channel_error_vs_old=float(np.abs(new[:256].astype(float)-old[:256]).mean()),gray256_max_channel_spread=int(np.ptp(new[:256].astype(int),axis=1).max()),rgb64_max_channel_error_vs_old=int(np.abs(new[256:].astype(int)-old[256:]).max()),coefficient_sha256=l.sha(l.D/'models'/e['id']/'coefficients-padded.bin'))
        rows.append(row);print('CONSTANT_CONTROL',row,flush=True)
    l.save_json(l.D/'constant-controls.json',dict(styles=rows,diagnostic_only_no_visual_acceptance=True,all_input_samples_are_constant_spatial_fields=True))
if __name__=='__main__':main()
