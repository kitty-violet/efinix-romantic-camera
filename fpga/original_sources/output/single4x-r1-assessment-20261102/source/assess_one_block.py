"""Read-only integer ablation of the existing S4R2 styles.

No checkpoint, accepted pack, operator implementation or hardware is modified.
Reduced graphs run existing integer coefficients in memory; not exported packs.
"""
from pathlib import Path
import sys,json,copy,hashlib,time,ast
import numpy as np
from PIL import Image

ROOT=Path(r'E:\fpga');BASE=ROOT/'output/single4x-r2-style-library-20261102';D=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'source'))
import library as l
from library import torch
sys.path.insert(0,str(ROOT/'output/single-down-up-reconstruction-20261102/source'))
from estimate_speed import bulk

def reduced(rows,drop):
    answer=[];replace={f'res{drop}_add':'down_pw' if drop==1 else 'res1_add'}
    for row in rows:
        if row['id'].startswith(f'res{drop}_'):continue
        row=copy.deepcopy(row);row['inputs']=[replace.get(x,x) for x in row['inputs']];answer.append(row)
    return answer

def compare(a,b):
    z=((a-b)*128).abs()
    return dict(maximum_channel_difference=int(z.max()),mean_absolute_difference=float(z.mean()),different_channel_values=int((z!=0).sum()),compared_channel_values=z.numel(),exact=bool(torch.equal(a,b)))

def variant(engine,drop,fold_b03=False):
    other=copy.copy(engine);other.rows=reduced(engine.rows,drop);other.ops=dict(engine.ops)
    if fold_b03:
        assert drop==2
        # A channelwise pure shift commutes with pointwise weak-edge synthesis.
        # Move each shift to mean/positive-detail/negative-detail lanes before
        # their final PW synthesis, then remove the otherwise-identity block.
        w,b,m=[x.clone() for x in engine.ops['res1_b_dw']]
        sw,sb,sm=engine.ops['res2_a_dw']
        for c in range(6):
            assert int((sw[c]!=0).sum())==1 and float(sb[0,c,0,0])==0
            assert float(sw[c].sum().double()*sm[0,c,0,0]/131072)==1
            for dst in [6+c,12+c,18+c]:
                w[dst]=sw[c];b[:,dst]=sb[:,c];m[:,dst]=sm[:,c]
        other.ops['res1_b_dw']=(w,b,m)
    return other

def inputs():
    records=[];images=[]
    src=l.OLD/'acceptance-four-scenes-5s-20261029/originals'
    for scene in ['indoor','outdoor','crowd','water']:
        p=src/f'{scene}_076.png';a=np.array(Image.open(p).convert('RGB'))
        for y,x in [(160,320),(360,800)]:
            images.append(a[y:y+256,x:x+256]);records.append(dict(scene=scene,source=str(p),sha256=l.sha(p),crop_y=y,crop_x=x,size=256))
    rng=np.random.default_rng(2026110209)
    for _ in range(2):images.append(rng.integers(0,256,(256,256,3),dtype=np.uint8))
    X=l.tx(np.stack(images))
    colours=np.array([[r,g,b] for r in [0,85,170,255] for g in [0,85,170,255] for b in [0,85,170,255]],np.uint8)
    flat=l.tx(np.broadcast_to(colours[:,None,None,:],(64,8,8,3)).copy())
    return X,flat,records

def cost_report(rows):
    newrows=reduced(rows,2);old=l.costs(rows);new=l.costs(newrows)
    assert old['parameters']==6787 and new['parameters']==4035
    assert old['macs_per_720p']==640051200 and new['macs_per_720p']==488908800
    ledger=json.loads((ROOT/'output/single-down-up-reconstruction-20261102/speed-estimate.json').read_text('utf-8'))
    oldbulk=bulk(rows);newbulk=bulk(newrows)
    previous=ledger['bulk_service_scenario'];time1=previous['new_calibrated_bulk_ms']*newbulk['cycles']/oldbulk['cycles']+previous['retained_other_ms']
    return dict(two_blocks=old,one_block=new,parameter_reduction_percent=100*(old['parameters']-new['parameters'])/old['parameters'],mac_reduction_percent=100*(old['macs_per_720p']-new['macs_per_720p'])/old['macs_per_720p'],ideal_mac_speedup=old['macs_per_720p']/new['macs_per_720p'],historical_150mhz_scenario=dict(two_block_ms=previous['conditional_frame_ms'],one_block_ms=time1,two_block_fps=1000/previous['conditional_frame_ms'],one_block_fps=1000/time1,saved_ms=previous['conditional_frame_ms']-time1,not_measured=True,do_not_add_other_workflow_savings=True),rows=newrows)

@torch.inference_mode()
def main():
    l.setup(20261102);l.guard();D.mkdir(parents=True,exist_ok=True);X,flat,records=inputs()
    entries=json.loads((BASE/'revisions/film-pixel-r2/inventory.json').read_text('utf-8'))['styles'];results=[];cost=None;start=time.perf_counter()
    for e in entries:
        key=e['id']
        if key=='BW_LINE':continue
        folder=(BASE/'revisions/film-pixel-r2/models'/key) if key in ['OLD_FILM','PIXEL_R39'] else BASE/'models'/key
        engine=l.PackedEngine(folder)
        if cost is None:cost=cost_report(engine.rows)
        baseline=engine(X);uniform=engine(flat);row=dict(id=key,name=e.get('current_display_name',e['name']),source_coefficients_sha256=l.sha(folder/'coefficients-padded.bin'),selected_source=str(folder))
        for drop in [1,2]:
            v=variant(engine,drop);out=v(X);f=v(flat)
            row[f'delete_block_{drop}']=dict(all_inputs=compare(out,baseline),flat_rgb64=compare(f,uniform),natural_8_patches=compare(out[:8],baseline[:8]),two_random_patches=compare(out[8:],baseline[8:]))
        if key=='B03':
            v=variant(engine,2,True);out=v(X);f=v(flat)
            row['move_shift_into_existing_first_block_then_delete_second']=dict(all_inputs=compare(out,baseline),flat_rgb64=compare(f,uniform))
        results.append(row);l.save_json(D/'progress.json',dict(completed=len(results),expected=16,last=key))
        print('ONE_BLOCK_ASSESS',key,'drop1',round(row['delete_block_1']['all_inputs']['mean_absolute_difference'],3),'drop2',round(row['delete_block_2']['all_inputs']['mean_absolute_difference'],3),'B03fold',row.get('move_shift_into_existing_first_block_then_delete_second'),flush=True)
        del engine,baseline,uniform,v,out,f;torch.cuda.empty_cache()
    exact=[r['id'] for r in results if r['delete_block_2']['all_inputs']['exact'] and r['delete_block_2']['flat_rgb64']['exact']]
    fold=[r['id'] for r in results if r.get('move_shift_into_existing_first_block_then_delete_second',{}).get('all_inputs',{}).get('exact') and r['move_shift_into_existing_first_block_then_delete_second']['flat_rgb64']['exact']]
    report=dict(status='READ_ONLY_INTEGER_ABLATION_COMPLETE_NOT_RETRAINED_OR_DEPLOYED',cost=cost,inputs=dict(source_patches=records,additional_random_patches=2,random_seed=2026110209,flat_rgb64=True,patches_are_assessment_not_training=True),results=results,delete_second_exact_in_all_tests=exact,move_shift_then_delete_exact_in_all_tests=fold,no_source_package_modified=True,no_hardware_modified=True,seconds=time.perf_counter()-start,limits=['Ablation error is not a lower bound on retrained one-block quality.','Finite test equality alone is not a universal proof; pure identity and pure-shift routes can additionally be inspected structurally.','The rejected large-square style route is not revived; fixed stride4/nearest4 phase is unchanged.'])
    l.save_json(D/'assessment.json',report)
    lines=['# 再减一个残差块：当前16款整数系数评估','','本次仅在内存删除节点运行整数系数，未训练、未导出新包、未改硬件。固定4倍采样不变，不恢复已否决的大方格路线。','','两残差6787参数/640051200MAC；一残差4035参数/488908800MAC。少4卷积与1残差合并节点，变成11卷积13节点。','','|编号|名称|删第一块MAE|删第二块MAE|删第二块逐值相同|','|---|---|---:|---:|---|']
    for r in results:lines.append(f"|{r['id']}|{r['name']}|{r['delete_block_1']['all_inputs']['mean_absolute_difference']:.3f}|{r['delete_block_2']['all_inputs']['mean_absolute_difference']:.3f}|{r['delete_block_2']['all_inputs']['exact']}|")
    lines+=['',f'直接删除第二块全部测试精确相同：{exact}',f'先把纯通道位移迁回第一块现有DW再删第二块，全部测试精确相同：{fold}','','比较包含四场景8个256方形块、2个固定种子随机块和64组平坦RGB；这是删层探针，不是重训结果，也不是完整视频验收。','',f"沿既有150MHz账本的相同假设，条件估计{cost['historical_150mhz_scenario']['two_block_fps']:.2f}到{cost['historical_150mhz_scenario']['one_block_fps']:.2f}FPS，少约{cost['historical_150mhz_scenario']['saved_ms']:.2f}ms/帧。未测量，不是当前其他工程的FPS，不与其他工作流节省时间相加。",'']
    (D/'评估结论.md').write_text('\n'.join(lines),encoding='utf-8');ast.parse(Path(__file__).read_text('utf-8'))
    print('ONE_BLOCK_ASSESSMENT_COMPLETE',exact,fold,flush=True)

if __name__=='__main__':main()
