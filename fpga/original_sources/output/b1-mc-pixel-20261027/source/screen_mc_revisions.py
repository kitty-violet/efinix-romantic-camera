"""Short consecutive-frame screen before further full-video renders."""
from core import *
import evaluation_utils as ev
from evaluate_pixel import mean_metrics
from network_ab.training.media import decode_frames
from network_ab.training.fpga_video_resize import area_3to2
from evaluate_r3_r4 import content
import argparse

@torch.inference_mode()
def main(args):
    setup();dest=OUT/args.dest;dest.mkdir(exist_ok=True);disk_guard();begin=time.perf_counter()
    paths=dict(soft_r4=OUT/'exports/mc-projector-soft-r4/coefficients-padded.bin',
        faces_r5=OUT/'r5-block-faces/candidate.bin',tones_r6=OUT/'r6-five-tones/initial.bin')
    if args.candidate:
        paths={}
        for item in args.candidate:
            name,path=item.split('=',1)
            assert name and name not in paths
            paths[name]=Path(path)
    engines={k:IntegerEngine(p.read_bytes()) for k,p in paths.items()}
    manifest=json.loads((ROOT/'output/b1-ink-wash-20261026/independent_test/manifest.json').read_text('utf-8'))
    reports=[]
    for clip in [c for c in manifest['clips'] if c['id'] in ['outdoor','station','dusk']]:
        records=[];prev=None;frames=[];original=[]
        old=json.loads((OUT/f'r3-r4-review/{clip["id"]}_report.json').read_text('utf-8'))
        byindex={r['frame']:r for r in old['records']}
        for index,stamp,full in decode_frames(clip['source'],seconds=1,hlg_exposure_stops=clip['exposure_stops']):
            if stamp>=1:break
            a=area_3to2(full);x=tensor(a)[None]*128;images=dict(source=a,**{k:rgb(n(x)/128) for k,n in engines.items()})
            frames.append(index);row=dict(frame=index,pts=stamp)
            if prev is not None and index%5==0:
                mx,my,valid,static,_=ev.flow_mask(prev['source'],a)
                for k,z in images.items():row[k]=ev.temporal(prev[k],z,mx,my,valid,static)
                assert abs(row['source']['warp_mae']-byindex[index]['source']['warp_mae'])<1e-5
                original.append(byindex[index])
            if index==0:
                for k in engines:Image.fromarray(images[k]).save(dest/f'{clip["id"]}_{k}.png')
                row['content']={k:content(a,images[k]) for k in engines}
            records.append(row);prev=images
        assert frames==list(range(len(frames))) and len(frames)>=29
        report=dict(clip=clip['id'],frame_count=len(frames),temporal={k:mean_metrics(records,k) for k in ['source',*engines]},
            previous={k:mean_metrics(original,k) for k in ['v14','flat_r3','mc_r4','mc_previous']},content=records[0]['content'])
        reports.append(report);save_json(dest/'summary.json',dict(reports=reports,coefficients={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()},
            full_video_validation=False,source_prefix='continuous original first1s, not sampled or retimed',elapsed=time.perf_counter()-begin))
        print('MC_SCREEN',clip['id'],json.dumps(report),flush=True)
    print('MC_SCREEN_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dest',default='mc-revision-screen')
    p.add_argument('--candidate',action='append',help='name=absolute BIN path; overrides default candidates')
    main(p.parse_args())
