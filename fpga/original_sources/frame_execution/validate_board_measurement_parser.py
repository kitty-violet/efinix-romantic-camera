"""Check frame identity, repeated-screen exclusion and incomplete-run rejection."""
import argparse,json
from pathlib import Path
from summarize_board_measurement import analyze
from run_units import ROOT,sha

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT/'tmp/frame-execution/board-measurement-parser1');args=parser.parse_args()
    lines=[]
    for frame in range(3):
        base=frame*40000000
        points=[('CAMERA_PORT_BEGIN',0),('RAW_FRAME_BEGIN',100),('RAW_FRAME_STORED',10000000),
                ('RGB_FRAME_BEGIN',11000000),('RAW_FRAME_READ_DONE',12000000),('RGB_FRAME_COMPLETE',13000000),
                ('CNN_BEGIN',14000000),('CNN_COMMIT',30000000),('SCREEN_BEGIN',31000000),('SCREEN_COMPLETE',32000000)]
        for event,offset in points:
            identity=f'camera={frame}' if event=='CAMERA_PORT_BEGIN' else f'frame={frame+1}'
            if event=='RAW_FRAME_BEGIN':identity+=f' camera={frame}'
            lines.append(f'{event} {identity} t_ns={base+offset}')
        lines.extend([f'SCREEN_BEGIN frame={frame+1} t_ns={base+33000000}',f'SCREEN_COMPLETE frame={frame+1} t_ns={base+34000000}'])
    result=analyze(lines)
    assert result['status']=='THREE_COMPLETE_FRAME_TIMESTAMPS'
    assert len(result['distinct_frame_intervals'])==2
    assert all(r['input_to_first_pixel_ms']==31 and r['input_to_last_pixel_ms']==32 for r in result['complete_frames'])
    assert all(r['new_screen_interval_ms']==40 and r['new_ai_interval_ms']==40 for r in result['distinct_frame_intervals'])
    assert all(abs(sum(x['duration_ms'] for x in r['exclusive_latency_ledger'])-32)<1e-9 and abs(r['unaccounted_latency_ms'])<1e-9 for r in result['complete_frames'])
    assert result['observed_rate']['ai_fps']==result['observed_rate']['new_screen_fps']==25
    assert result['frame_counts']==dict(camera_port=3,raw_capture_started=3,cnn_committed=3,causally_complete_screen=3)
    skipped=analyze(lines+['CNN_COMMIT frame=4 t_ns=50000000','CAMERA_PORT_BEGIN camera=9 t_ns=120000000'])
    assert skipped['observed_rate']['ai_fps']==37.5 and skipped['observed_rate']['new_screen_fps']==25
    assert skipped['frame_counts']['cnn_committed']==4 and skipped['unbound_camera_frames']==[9]
    for bad in (lines+['CAMERA_PORT_BEGIN camera=0 t_ns=1'],lines+['RAW_FRAME_BEGIN frame=4 camera=0 t_ns=5']):
        try:analyze(bad)
        except ValueError:pass
        else:raise AssertionError('Repeated camera identity accepted')
    partial=analyze([x for x in lines if not x.startswith('SCREEN_COMPLETE')])
    assert not partial['complete_frames'] and partial['status']=='INCOMPLETE_FRAME_TIMESTAMPS'
    rejected=0
    for bad in [lines+['CNN_COMMIT frame=1 t_ns=31000000'],[x.replace('RAW_FRAME_BEGIN frame=1 camera=0','RAW_FRAME_BEGIN frame=1 camera=2') for x in lines]]:
        try:analyze(bad)
        except ValueError:rejected+=1
    assert rejected==2
    overlap=analyze([x.replace('RGB_FRAME_COMPLETE frame=1 t_ns=13000000','RGB_FRAME_COMPLETE frame=1 t_ns=15000000') for x in lines])
    assert overlap['complete_frames'][0]['rgb_complete_to_cnn_begin_ms']==-1
    assert sum(x['duration_ms'] for x in overlap['complete_frames'][0]['exclusive_latency_ledger'])==32
    try:analyze([x.replace('RGB_FRAME_COMPLETE frame=1 t_ns=13000000','RGB_FRAME_COMPLETE frame=1 t_ns=5000000') for x in lines])
    except ValueError:pass
    else:raise AssertionError('Reversed RGB span accepted')
    old=ROOT/'tmp/frame-execution/board-questa-direct3/run_result.json'
    assert analyze(json.loads(old.read_text())['events'])['complete_frames']==[]
    root=args.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(exist_ok=False)
    report=dict(status='FRAME_MEASUREMENT_PARSER_PASS',checks=['three distinct frames','repeat scans excluded','incomplete screen rejected','duplicate compute rejected','wrong camera identity rejected','actual early failure produces no frame time','exclusive ledger includes all latency','observed FPS uses distinct intervals','overlap not double-counted','reversed RGB span rejected','all commits counted despite undisplayed result','unbound camera inputs retained','duplicate camera and capture identities rejected'],
                parser_sha256=sha(Path(__file__).with_name('summarize_board_measurement.py')),prior_run_sha256=sha(old),board_measurement=False)
    (root/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
