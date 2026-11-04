"""One controlled route with the measured PLL-ratio correction, same placement."""
import json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import once
from camera_checked_v2 import BOARD
import camera_checked_board
camera_checked_board.BOARD=BOARD

BASE=ROOT/'tmp/frame-execution'
PHYSICAL=BASE/'camera-checked-bound3'

def prepare():
    from camera_config_clock_ratio import transform
    from prepare_local_driver_route import main as old_prepare
    import prepare_local_driver_route as old
    review_path=BASE/'camera-checked-route-review3.json';assert not review_path.exists()
    r=json.loads((BASE/'camera-checked-route-review2.json').read_text())
    r['physical']=str(PHYSICAL)
    r['rationale'] += [
        'The first route was legal but core setup -0.910ns, dominated by a 5.213ns local_driver_net_094 wire into CNN RAM. The new map/placement remains unchanged; no seed sweep or unproved cell relocation.',
        'PLL4 out2/out1 use exact dividers4/120 at phase0. Existing 6.666/200.000ns rounding creates a fictitious 0.002ns capture window. Same-route probe2 changes only CLK_5M to199.980ns: configuration setup improves -0.700 to+5.964ns; core stays-0.910ns. This removes false routing pressure without relaxing any period or adding exceptions.',
        'One TIMING_1 reroute on the reviewed C2 placement isolates that correction. No claim that changing recipe or repeating route guarantees success. The rejected broader pixel-clock edit is excluded.',
        'Existing failed-route reports now cover all95 held payload paths (maximum0.995ns),70DDR Gray bits,6clock observations and3retained legacy probe crossings. Fresh route must satisfy the same checks.'
    ]
    for f in (BASE/'camera-related-clock-sta2/result.json',BASE/'camera-related-clock-sta2/inputs.json',
              BASE/'camera-failed-route-gray1/camera_inventory_reaudit.json',BASE/'camera-checked-route2/camera_implementation_reaudit.json',
              HERE/'camera_config_clock_ratio.py',HERE/'camera_checked_cdc_v2.py',HERE/'camera_checked_gray_v2.py',
              HERE/'audit_camera_implementation.py',Path(__file__).resolve()):r['reviewed_evidence'][str(f)]=sha(f)
    review_path.write_text(json.dumps(r,indent=2))
    old.audit_delta=camera_checked_board.audit_delta
    sys.argv=[str(HERE/'prepare_local_driver_route.py'),'--review',str(review_path)];old_prepare()
    sdc=PHYSICAL/'ti60f225_oob.sdc';before=sha(sdc)
    new,change=transform(sdc.read_text(),(PHYSICAL/'ti60f225_oob.peri.xml').read_text());sdc.write_text(new,encoding='utf8')
    record=dict(status='EXACT_CONFIG_CORE_PLL_RATIO_MODEL',change=change,before_sdc_sha256=before,
        after_sdc_sha256=sha(sdc),transform_sha256=sha(HERE/'camera_config_clock_ratio.py'),
        peri_sha256=sha(PHYSICAL/'ti60f225_oob.peri.xml'),hardware_access=False,board_release=False)
    model=PHYSICAL/'config_clock_ratio.json';model.write_text(json.dumps(record,indent=2))
    meta_file=PHYSICAL/'physical_inputs.json';meta=json.loads(meta_file.read_text())
    for row in meta['inputs']:
        if Path(row['snapshot'])==sdc:row['sha256']=sha(sdc)
    meta['clock_profile']['before_config_ratio_sha256']=meta['clock_profile']['after_sha256']
    meta['clock_profile']['after_sha256']=sha(sdc);meta['config_clock_ratio_sha256']=sha(model)
    meta_file.write_text(json.dumps(meta,indent=2))
    r=json.loads((PHYSICAL/'route_review.json').read_text())
    r['required_evidence'][str(meta_file)]=sha(meta_file);r['required_evidence'][str(model)]=sha(model)
    r['required_evidence'][str(sdc)]=sha(sdc)
    (PHYSICAL/'route_review.json').write_text(json.dumps(r,indent=2))
    print('SAME_MAP_PLACEMENT_CONFIG_RATIO_ROUTE_READY')

def route():
    from route_local_driver_board import driver_code
    from camera_checked_flow import generated
    source,code=driver_code()
    code=once(code,'from local_driver_board import audit_delta','from camera_checked_board import audit_delta')
    code=once(code,'import audit_control_state_implementation as impl','import audit_camera_implementation as impl')
    code=once(code,"HERE/'audit_native_split_gray.py'","HERE/'camera_checked_gray_v2.py'")
    generated(source,code,'-local-driver-route')

if __name__=='__main__':
    action=sys.argv.pop(1)
    if action=='prepare':prepare()
    elif action=='route':route()
    else:raise ValueError(action)
