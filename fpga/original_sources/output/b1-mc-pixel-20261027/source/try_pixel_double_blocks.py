"""One final fixed-B1 coarsening attempt; explicitly audit strict 8x8 failure.

A translation-equivariant stride-4 graph cannot implement an absolute stride-8
lattice for arbitrary interiors. Majority grouping is an approximation only.
"""
from core import *
from evaluate_r3_r4 import controls,content

D=OUT/'pixel-r40-double-blocks'
BASE=OUT/'exports/pixel-A-sharp-medium-r39'

def grid_stats(z,cell):
    h,w=z.shape[:2];v=z.reshape(h//cell,cell,w//cell,cell,3)
    diff=v.max((1,3)).astype(int)-v.min((1,3)).astype(int)
    return dict(cell=cell,nonuniform_cell_fraction=float(np.any(diff>0,2).mean()),max_in_cell_channel_range=int(diff.max()))

def regions(z):
    # Same exact RGB palette runs, measured on the physical 4px lattice.
    q=z[::4,::4];runs=[]
    for v in [q,q.transpose(1,0,2)]:
        for row in v:
            changes=np.where(np.any(row[1:]!=row[:-1],1))[0]+1
            runs.extend(np.diff(np.r_[0,changes,len(row)]).tolist())
    return dict(mean_run_pixels=float(np.mean(runs)*4),median_run_pixels=float(np.median(runs)*4),
                fraction_runs_at_least_8px=float((np.array(runs)>=2).mean()))

def reference_8(z):
    # Offline comparison only. Select a nearest existing colour in each 8x8 tile.
    low=z[::4,::4];h,w=low.shape[:2]
    cells=low.reshape(h//2,2,w//2,2,3).transpose(0,2,1,3,4).reshape(h//2,w//2,4,3)
    mean=cells.astype(float).mean(2);index=((cells-mean[:,:,None])**2).sum(3).argmin(2)
    picked=np.take_along_axis(cells,index[:,:,None,None],axis=2)[:,:,0]
    return picked.repeat(8,0).repeat(8,1)

@torch.no_grad()
def make():
    n=load_model(BASE/'selected.pt');s=131071/131072
    # Two existing depthwise kernels, symmetric binary-majority grouping.
    # No spatial phase, no resizing, no new operator, no change of the Y curve.
    for key in ['res2_conv2__dw','res3_conv2__dw']:
        n.weight[key][:24].fill_(127*s);n.bias[key][:24]=-571*127*s
        getattr(n,key+'_m')[:,:24]=131071
    return n

@torch.inference_mode()
def main():
    setup();disk_guard();D.mkdir(exist_ok=True);shutil.copy2(__file__,D/'source_used.py')
    n=make();raw=n.export();e=IntegerEngine(raw);base=IntegerEngine((BASE/'coefficients-padded.bin').read_bytes())
    (D/'candidate.bin').write_bytes(raw)
    torch.save(dict(model=n.state_dict(),residual_encoding=copy.deepcopy(n.package['residual_encoding']),
        qat=True,step=0,trained=False,source=str(BASE/'selected.pt'),attempt='one final content-guided coarsening',
        strict8_grid=False),D/'constructed.pt')
    c=controls({'base':base,'candidate':e})
    assert c['base']['gray_curve_mean']==c['candidate']['gray_curve_mean']
    assert c['candidate']['gray_sweep_max_channel_difference']==0
    cases={'corridor':OUT/'data-quality/VID_20260924_231158_source.png','outdoor':OUT/'targets/outdoor_source.png',
           'indoor_training':OUT/'data-quality/VID_20260925_070939_source.png'}
    rows=[]
    for scene,p in cases.items():
        a=read(p);x=tensor(a)[None]*128;b=rgb(base(x)/128);z=rgb(e(x)/128);target=reference_8(b)
        assert torch.equal(n(x/128)*128,e(x));assert grid_stats(z,4)['nonuniform_cell_fraction']==0
        assert grid_stats(target,8)['nonuniform_cell_fraction']==0
        # Interior translation audit: 4px shift commutes with the fixed graph.
        shifted=np.pad(a,((0,0),(4,0),(0,0)),mode='edge')[:,:-4]
        shifted_out=rgb(e(tensor(shifted)[None]*128)/128)
        commute_error=int(np.abs(shifted_out[80:-80,84:-80].astype(int)-z[80:-80,80:-84]).max());assert commute_error==0
        r=dict(scene=scene,base_grid4=grid_stats(b,4),base_grid8=grid_stats(b,8),
            candidate_grid4=grid_stats(z,4),candidate_grid8=grid_stats(z,8),
            reference_grid8=grid_stats(target,8),base_regions=regions(b),candidate_regions=regions(z),
            content=content(a,z),base_content=content(a,b),translation4_interior_max_error=commute_error,
            strict8_succeeded=grid_stats(z,8)['nonuniform_cell_fraction']==0)
        rows.append(r)
        for name,v in [('base',b),('candidate',z),('reference8_not_B1',target)]:Image.fromarray(v).save(D/f'{scene}_{name}.png')
        board([[b,target,z]],D/f'{scene}_comparison.jpg',
            ['已选R39中强 / 实际4×4','严格8×8示意 / 非B1输出','最后合并尝试 / 实际B1仍4网格'],width=640)
        print('DOUBLE_BLOCK',json.dumps(r),flush=True)
    result=dict(single_final_attempt=True,base_user_selected=True,user_selected_sharpness='medium',
        original_native_cell=4,requested_native_cell=8,fixed_topology=True,training=False,records=rows,
        controls=c,strict8_succeeded=all(r['strict8_succeeded'] for r in rows),
        reason='Only two stride2 stages; stride4 translation equivariance in the interior. Coarsening is not a strict8 lattice.',disk=disk_guard())
    save_json(D/'summary.json',result)
    selected_path=OUT/'SELECTED_PIXEL_CURVE.json';selected=json.loads(selected_path.read_text('utf-8-sig'))
    selected.update(user_selected_sharpness='medium',user_selected_pixel_export=str(BASE),
        preferred_spatial_export=str(BASE),latest_preview_export=str(BASE),latest_preview_name=BASE.name,
        latest_preview_approved=True,selected_medium_crc32='2d9aad69',deployment_changed=False,
        final_block_doubling_attempt=str(D),future_pixel_scope='User chose R39 medium. One final doubling attempt only; preserve choice unless user accepts the trial. MC closed.')
    save_json(selected_path,selected)

if __name__=='__main__':main()
