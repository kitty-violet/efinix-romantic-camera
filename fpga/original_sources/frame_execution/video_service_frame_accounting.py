"""Offline accounting for a supervised, drained LIVE_A batch.

No USB access. Counts are taken at stopped/drained boundaries, so asynchronously
read process words remain stable. Physical refresh and camera sticky bits must
not be passed off as successful input/output frame counts.
"""
import argparse,json
from pathlib import Path

COUNTERS={
 'captured':(16,'wrap','RGB frame admission, not successful capture'),
 'skipped':(16,'wrap','RGB SOF deliberately rejected; includes stopped service'),
 'orphan_pixels':(16,'saturate','Pixels without a selected frame, not frames'),
 'compute_admitted':(16,'saturate','Accepted CNN starts'),
 'committed':(32,'wrap','Successful AI result commits in LIVE_A'),
 'dropped_results':(16,'wrap','Completed results discarded in favor of a newer display front'),
 'repeated_reads':(16,'wrap','DDR reads of retained front; not physical refresh count'),
 'source_area_frames':(16,'saturate','Downscaled SOF handshakes, not camera frame starts'),
 'physical_frames':(16,'wrap','HDMI timing frame wraps'),
 'complete_frames':(16,'wrap','Complete source frames consumed by display'),
 'distinct_frames':(16,'wrap','Completed display frames with a changed frame ID'),
 'repeated_frames':(16,'wrap','Completed display frames with unchanged frame ID'),
}

def delta(name,before,after,maximum_events):
    bits,policy,_=COUNTERS[name];limit=(1<<bits)-1
    if any(type(x) is not int or not 0<=x<=limit for x in (before,after)):
        raise ValueError('Counter value outside actual hardware width: '+name)
    if type(maximum_events) is not int or not 0<=maximum_events<1<<bits:
        raise ValueError('A sub-wrap event bound is required: '+name)
    if policy=='saturate' and (before==limit or after==limit):
        raise ValueError('Saturated counter is a lower bound, not an exact measurement: '+name)
    if policy=='saturate' and after<before:raise ValueError('Counter reset/discontinuity: '+name)
    value=(after-before)&limit
    if value>maximum_events:raise ValueError('Counter discontinuity or exceeded event bound: '+name)
    return value

def drained_batch(before,after,*,maximum_events=4096):
    for s in (before,after):
        if type(s['reset_epoch']) is not int or not 0<=s['reset_epoch']<65535:
            raise ValueError('Saturated/invalid reset history cannot exclude an intervening reset')
        if s['phase']!=5 or s['mode']!='LIVE_A' or not s['work_drained']:
            raise ValueError('A stopped, prepared, drained LIVE_A boundary is required')
        if s['apb_errors'] or s['owner_error'] or s['control_errors']:
            raise ValueError('Faulted batch cannot use the clean owner balance')
        if not s['stable_process_counts']:
            raise ValueError('Accepted/start/commit counts must be read twice and stable')
    if before['reset_epoch']!=after['reset_epoch'] or before['bitstream_sha256']!=after['bitstream_sha256']:
        raise ValueError('Reset or reconfiguration separates the observations')
    d={name:delta(name,before[name],after[name],maximum_events) for name in ('captured','compute_admitted','committed')}
    if not d['captured']>=d['compute_admitted']>=d['committed']:
        raise ValueError('Conservation violation; inspect ownership and measurement semantics')
    capture_failed=d['captured']-d['compute_admitted']
    compute_failed=d['compute_admitted']-d['committed']
    return dict(status='DRAINED_LIVE_A_BATCH_ACCOUNTED',accepted_input_frames=d['captured'],
        successful_capture_frames=d['compute_admitted'],capture_failed_after_admission=capture_failed,
        cnn_started=d['compute_admitted'],cnn_failed=compute_failed,cnn_completed=d['committed'],
        admitted_frame_failure_fraction=(capture_failed+compute_failed)/d['captured'] if d['captured'] else None,
        camera_input_frames=None,raw_failures_before_admission=None,intentional_run_time_skips=None,
        screen_new_frames=None,screen_repeated_frames=None,
        notes=['No pending CAPTURE/CAPTURED/COMPUTE slot at either boundary.',
               'Failure fraction denominator is admitted RGB frames, not all camera frames.',
               'RAW sticky flags are evidence of occurrence, not event counts.',
               'No FPS is inferred from batch counts without a separately timed run interval.',
               'Missing measurements are null, never zero.'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('Do not overwrite evidence')
    result=drained_batch(json.loads(a.before.read_text(encoding='utf-8')),json.loads(a.after.read_text(encoding='utf-8')))
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
