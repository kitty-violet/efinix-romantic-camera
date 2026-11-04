"""Join actual same-frame boundaries; refuse to manufacture missing frame times."""
import argparse,json,re
from pathlib import Path
from run_units import sha

def analyze(lines):
    events=[]
    for line in lines:
        match=re.search(r'\b(CAMERA_PORT_BEGIN|RAW_FRAME_BEGIN|RAW_FRAME_STORED|RAW_FRAME_READ_DONE|RGB_FRAME_BEGIN|RGB_FRAME_COMPLETE|CNN_BEGIN|CNN_COMMIT|SCREEN_BEGIN|SCREEN_COMPLETE|BOARD_CPU_RUN|BOARD_CAL_DONE|PAYLOAD_PRELOADED)\b(.*)',line)
        if not match:continue
        fields=dict(re.findall(r'(\w+)=([^\s,]+)',match[2]))
        if 't_ns' not in fields:raise ValueError('Missing event time: '+line)
        events.append(dict(kind=match[1],time_ns=float(fields['t_ns']),fields=fields))
    cameras={};frames={};startup={}
    for event in events:
        kind,t,f=event['kind'],event['time_ns'],event['fields']
        if kind=='CAMERA_PORT_BEGIN':
            camera=int(f['camera'])
            if camera in cameras:raise ValueError(f'Duplicate camera frame {camera}')
            cameras[camera]=t
        elif 'frame' in f:
            row=frames.setdefault(int(f['frame']),{})
            # First screen appearance is latency; repeated scans are not new AI frames.
            if kind in row and kind not in ('SCREEN_BEGIN','SCREEN_COMPLETE'):
                raise ValueError(f'Duplicate {kind} for frame {f["frame"]}')
            row.setdefault(kind,t)
            if kind=='RAW_FRAME_BEGIN':row['camera']=int(f['camera'])
        else:startup[kind]=t
    bindings={}
    for fid,row in frames.items():
        if 'camera' in row:
            if row['camera'] in bindings:raise ValueError('Camera frame captured twice')
            bindings[row['camera']]=fid
    complete=[];incomplete=[]
    for fid,row in frames.items():
        required=['camera','RAW_FRAME_BEGIN','RAW_FRAME_STORED','RAW_FRAME_READ_DONE','RGB_FRAME_BEGIN','RGB_FRAME_COMPLETE','CNN_BEGIN','CNN_COMMIT','SCREEN_BEGIN','SCREEN_COMPLETE']
        absent=[key for key in required if key not in row]
        if 'camera' in row and row['camera'] not in cameras:absent.append('CAMERA_PORT_BEGIN')
        if absent:incomplete.append(dict(frame=fid,missing=absent));continue
        start=cameras[row['camera']]
        # RAW read and RGB conversion overlap; spans must not be summed as exclusive costs.
        bounds=[start,row['RAW_FRAME_BEGIN'],row['RAW_FRAME_STORED'],row['RGB_FRAME_BEGIN'],row['CNN_BEGIN'],row['CNN_COMMIT'],row['SCREEN_BEGIN'],row['SCREEN_COMPLETE']]
        if any(b<a for a,b in zip(bounds,bounds[1:])):raise ValueError(f'Non-causal frame association {fid}: {bounds}')
        if not row['RGB_FRAME_BEGIN']<=row['RGB_FRAME_COMPLETE']<=row['SCREEN_COMPLETE']:
            raise ValueError(f'Non-causal RGB completion {fid}')
        if not row['RAW_FRAME_STORED']<=row['RAW_FRAME_READ_DONE']<=row['SCREEN_COMPLETE']:
            raise ValueError(f'Non-causal RAW completion {fid}')
        # This ledger is exhaustive and non-overlapping. The internal RAW,
        # RGB and CNN spans are diagnostic overlays, never added twice.
        endpoints=[('camera_capture_and_raw_store',start,row['RAW_FRAME_STORED']),
                   ('raw_read_preprocess_and_compute_admission',row['RAW_FRAME_STORED'],row['CNN_BEGIN']),
                   ('cnn_execution_and_result_commit',row['CNN_BEGIN'],row['CNN_COMMIT']),
                   ('wait_for_first_screen_pixel',row['CNN_COMMIT'],row['SCREEN_BEGIN']),
                   ('screen_active_scan',row['SCREEN_BEGIN'],row['SCREEN_COMPLETE'])]
        ledger=[dict(stage=name,start_ns=begin,end_ns=end,duration_ms=(end-begin)/1e6) for name,begin,end in endpoints]
        total=(row['SCREEN_COMPLETE']-start)/1e6
        uncovered=total-sum(x['duration_ms'] for x in ledger)
        if abs(uncovered)>1e-9:raise ValueError(f'Incomplete time ledger {fid}: {uncovered}')
        complete.append(dict(frame=fid,camera=row['camera'],camera_port_ns=start,
            capture_store_ms=(row['RAW_FRAME_STORED']-start)/1e6,
            raw_read_rgb_span_ms=(row['RGB_FRAME_COMPLETE']-row['RAW_FRAME_STORED'])/1e6,
            cnn_compute_commit_ms=(row['CNN_COMMIT']-row['CNN_BEGIN'])/1e6,
            commit_to_screen_ms=(row['SCREEN_BEGIN']-row['CNN_COMMIT'])/1e6,
            screen_scan_ms=(row['SCREEN_COMPLETE']-row['SCREEN_BEGIN'])/1e6,
            input_to_first_pixel_ms=(row['SCREEN_BEGIN']-start)/1e6,
            input_to_last_pixel_ms=(row['SCREEN_COMPLETE']-start)/1e6,
            cnn_commit_ns=row['CNN_COMMIT'],screen_begin_ns=row['SCREEN_BEGIN'],
            observed_boundaries_ns={k:v for k,v in row.items() if k!='camera'},
            exclusive_latency_ledger=ledger,unaccounted_latency_ms=uncovered,
            rgb_complete_to_cnn_begin_ms=(row['CNN_BEGIN']-row['RGB_FRAME_COMPLETE'])/1e6))
    complete.sort(key=lambda row:row['screen_begin_ns'])
    intervals=[dict(previous=a['frame'],frame=b['frame'],new_screen_interval_ms=(b['screen_begin_ns']-a['screen_begin_ns'])/1e6,
                    new_ai_interval_ms=(b['cnn_commit_ns']-a['cnn_commit_ns'])/1e6) for a,b in zip(complete,complete[1:])]
    if any(x['new_ai_interval_ms']<=0 or x['new_screen_interval_ms']<=0 for x in intervals):
        raise ValueError('Non-positive distinct-frame interval')
    # Computing and displaying are separate events. An undisplayed result must
    # not disappear from AI FPS merely because it lacks a screen completion.
    commits=sorted((row['CNN_COMMIT'],fid) for fid,row in frames.items() if 'CNN_COMMIT' in row)
    ai_intervals=[dict(previous=a[1],frame=b[1],interval_ms=(b[0]-a[0])/1e6) for a,b in zip(commits,commits[1:])]
    if any(x['interval_ms']<=0 for x in ai_intervals):raise ValueError('Non-positive AI commit interval')
    source=sorted((t,camera) for camera,t in cameras.items())
    camera_intervals=[dict(previous=a[1],camera=b[1],interval_ms=(b[0]-a[0])/1e6) for a,b in zip(source,source[1:])]
    if any(x['interval_ms']<=0 for x in camera_intervals):raise ValueError('Non-positive camera interval')
    observed_rate=None
    if intervals or ai_intervals:
        ai_mean=sum(x['interval_ms'] for x in ai_intervals)/len(ai_intervals) if ai_intervals else None
        screen_mean=sum(x['new_screen_interval_ms'] for x in intervals)/len(intervals) if intervals else None
        observed_rate=dict(interval_count=len(intervals),ai_interval_count=len(ai_intervals),ai_mean_interval_ms=ai_mean,
                           ai_fps=1000/ai_mean if ai_mean else None,new_screen_mean_interval_ms=screen_mean,
                           new_screen_fps=1000/screen_mean if screen_mean else None)
    return dict(status='THREE_COMPLETE_FRAME_TIMESTAMPS' if len(complete)>=3 else 'INCOMPLETE_FRAME_TIMESTAMPS',
        complete_frames=complete,distinct_frame_intervals=intervals,observed_rate=observed_rate,incomplete_frames=incomplete,startup_ns=startup,
        all_ai_commit_intervals=ai_intervals,camera_intervals=camera_intervals,
        frame_counts=dict(camera_port=len(cameras),raw_capture_started=len(bindings),cnn_committed=len(commits),
                          causally_complete_screen=len(complete)),
        unbound_camera_frames=[camera for _,camera in source if camera not in bindings],
        note='Observed intervals only. RAW/RGB/CNN spans may overlap. No STA or fit inference.')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    root=args.root.resolve();source=root/'run_result.json';run=json.loads(source.read_text())
    result=analyze(run['events']);result['run_sha256']=sha(source)
    result['inputs_sha256']=run['inputs_sha256'];result['ddr_model_error_total']=run['ddr_model_error_total']
    result['measurement_valid']=len(result['complete_frames'])>=3 and not run['failures'] and run['returncode']==0 and not run['ddr_model_error_total']
    (root/'frame_measurement.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
