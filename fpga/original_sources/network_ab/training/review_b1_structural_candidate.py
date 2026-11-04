"""Paired structural review, with no scalar trade-off hiding regressions.

These are engineering screening limits, not a substitute for visual/video
acceptance. Read unencoded native720 PNG audits from the two specialized
measurement tools. Exported-package equivalence is checked separately.
"""
import argparse,json
from pathlib import Path
from .runtime import write_json,file_hash


def assess(edge,narrow,baseline,candidate):
    e0,e1,et=[edge['summary'][k] for k in (baseline,candidate,'target')]
    n0,n1=[narrow['summary'][k] for k in (baseline,candidate)]
    checks={
        'edge_width_improved':e1['width_p50']<e0['width_p50'],
        'narrow_contrast_improved':n1['median_contrast']>n0['median_contrast'],
        'lost_narrow_fraction_not_worse':n1['fraction_below_half']<=n0['fraction_below_half'],
        'weak_step_fraction_not_worse':e1['low_contrast_fraction']<=e0['low_contrast_fraction']+.005,
        'edge_colour_error_not_worse':e1['off_segment_rgb_rms_mean']<=e0['off_segment_rgb_rms_mean']*1.05,
        'narrow_colour_error_not_worse':n1['chromatic_direction_error']<=n0['chromatic_direction_error']*1.05,
        'narrow_outside_ridge_not_worse':n1['outside_ridge_error']<=n0['outside_ridge_error']*1.05,
    }
    limits={
        'median_edge_at_most_target_plus_one_pixel':e1['width_p50']<=et['width_p50']+1,
        'median_narrow_contrast_at_least_70_percent':n1['median_contrast']>=.70,
        'narrow_below_half_at_most_20_percent':n1['fraction_below_half']<=.20,
    }
    return dict(paired_regression_checks=checks,engineering_screening_limits=limits,
        all_paired_checks_pass=all(checks.values()),all_numeric_screens_pass=all(checks.values()) and all(limits.values()),
        visual_acceptance=False,video_acceptance=False,automatic_release=False,
        note='Thresholds are explicit engineering screens. Passing never proves material style, corners, diagonal branches, colour, temporal stability, or hardware timing.')


def regression():
    e={'summary':{'target':dict(width_p50=2.),'before':dict(width_p50=6.,low_contrast_fraction=.08,off_segment_rgb_rms_mean=4.),
        'after':dict(width_p50=7.,low_contrast_fraction=.06,off_segment_rgb_rms_mean=3.)}}
    n={'summary':{'before':dict(median_contrast=.3,fraction_below_half=.7,chromatic_direction_error=3.,outside_ridge_error=.2),
        'after':dict(median_contrast=.2,fraction_below_half=.8,chromatic_direction_error=2.,outside_ridge_error=.1)}}
    r=assess(e,n,'before','after');assert not r['all_paired_checks_pass'] and not r['automatic_release']
    print('STRUCTURAL_SCREEN_PASS colour_improvement_cannot_hide_blur_or_missing_lines',flush=True)


def main():
    ap=argparse.ArgumentParser()
    for k in ('edge','narrow','baseline','candidate','out'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();regression();edge=json.loads(Path(a.edge).read_text('utf-8'));narrow=json.loads(Path(a.narrow).read_text('utf-8'))
    for obj,field in ((edge,'edges'),(narrow,'records')):
        for row in obj['records']:
            for sample in row[field]:
                assert all(k in sample['values'] for k in ('target',a.baseline,a.candidate))
    assert [r['id'] for r in edge['records']]==[r['id'] for r in narrow['records']]
    for obj in (edge,narrow):
        assert all(file_hash(p)==sha for p,sha in obj['png_sha256'].items())
    report=assess(edge,narrow,a.baseline,a.candidate)
    report.update(baseline=a.baseline,candidate=a.candidate,edge_sha256=file_hash(a.edge),narrow_sha256=file_hash(a.narrow))
    dest=Path(a.out);assert not dest.exists();write_json(dest,report);print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
