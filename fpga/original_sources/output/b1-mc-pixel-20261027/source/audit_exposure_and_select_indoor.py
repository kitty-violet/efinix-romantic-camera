"""Local data selection only; preserve all original assets and manifests."""
from core import *
from network_ab.training.media import decode_frames
from network_ab.training.fpga_video_resize import area_3to2

def metrics(a):
    y=a.astype(np.float32)@np.array([.2126,.7152,.0722],np.float32)
    white=(a.min(2)>=245).astype(np.uint8)
    _,_,stats,_=cv2.connectedComponentsWithStats(white,connectivity=4)
    largest=float(stats[1:,cv2.CC_STAT_AREA].max()/white.size) if len(stats)>1 else 0.
    return dict(mean_luma=float(y.mean()),p01=float(np.percentile(y,1)),p50=float(np.percentile(y,50)),
        p99=float(np.percentile(y,99)),near_white_rgb245_fraction=float(white.mean()),
        largest_near_white_component_fraction=largest,dark_below10_fraction=float((y<10).mean()),
        saturation_risk=bool(white.mean()>.10 or largest>.04))

def main():
    setup(271161);d=OUT/'data-quality';d.mkdir(exist_ok=True);disk_guard()
    audit=json.loads((ROOT/'output/b1-ink-wash-20261026/audit.json').read_text('utf-8'))
    train=[]
    for r in audit['train_records']:
        m=metrics(read(r['path']));train.append(dict(**r,exposure=m,exclude_from_future_local_fit=m['saturation_risk']))
    save_json(d/'training-exposure-audit.json',dict(records=train,thresholds=dict(near_white_rgb_min=245,total_fraction=.10,largest_component_fraction=.04),
        scope='only this workflow; diagnostic near-white clipping proxy, not a sensor RAW clipping measurement',
        original_data_unchanged=True,source_audit_sha256=sha(ROOT/'output/b1-ink-wash-20261026/audit.json')))
    # Review all content sources at three fixed times, independent of style output.
    incoming=WORK/'data/incoming/content/VID_20260924_231158等16项文件'
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14);rows=[];thumbs=[]
    for p in sorted(incoming.glob('*.mp4')):
        observations=[];first=None
        for index,stamp,a in decode_frames(p,indices=[0,45,90],hlg_exposure_stops=-1):
            a=area_3to2(a);observations.append(dict(frame=index,pts=stamp,**metrics(a)))
            if first is None:first=a
        Image.fromarray(first).save(d/(p.stem+'_source.png'))
        tile=Image.new('RGB',(384,256),'#111722');tile.paste(Image.fromarray(first).resize((384,216),Image.Resampling.LANCZOS),(0,40))
        draw=ImageDraw.Draw(tile);m=observations[0]
        draw.text((4,0),p.stem,font=font,fill='white')
        draw.text((4,19),f"Y={m['mean_luma']:.0f}  白区={100*m['near_white_rgb245_fraction']:.2f}%",font=font,fill='white')
        thumbs.append(tile);rows.append(dict(source=str(p),source_sha256=sha(p),observations=observations))
        print('EXPOSURE_SOURCE_REVIEW',p.name,round(m['mean_luma'],1),round(m['near_white_rgb245_fraction']*100,2),flush=True)
    sheet=Image.new('RGB',(384*4,256*((len(thumbs)+3)//4)),'#111722')
    for i,tile in enumerate(thumbs):sheet.paste(tile,(i%4*384,i//4*256))
    sheet.save(d/'source_contact_sheet.jpg',quality=94)
    save_json(d/'source-exposure-audit.json',dict(sources=rows,exposure_stops=-1,decode='existing fixed HLG-to-SDR policy',
        colour_or_exposure_adaptation=False,selection_based_on_source_only=True))
    print('TRAIN_EXPOSURE_AUDIT',len(train),'excluded',sum(r['exclude_from_future_local_fit'] for r in train),flush=True)
    # Remove only the obsolete station-render files from our interrupted job.
    cache=OUT/'cache/latest-mc-station'
    if cache.exists():
        resolved=cache.resolve();assert resolved.parent==(OUT/'cache').resolve() and not cache.is_symlink()
        owned=list(cache.iterdir())
        assert all(p.is_file() and p.resolve().parent==resolved and p.name.startswith(('native_','comparison_','station_')) and p.suffix in ['.png','.ffconcat'] for p in owned)
        for p in owned:p.unlink()
        resolved.rmdir()
    print('EXPOSURE_AUDIT_DONE',disk_guard(),flush=True)

if __name__=='__main__':main()
