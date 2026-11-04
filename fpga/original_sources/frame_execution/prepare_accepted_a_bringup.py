"""Create a bounded commissioning bundle only from verified frozen evidence."""
import argparse
import shutil
from pathlib import Path
from accepted_a_assets import ROOT, HERE, ACCEPTED, PACKAGE_SHA, read_json, write_json, sha, accepted_package
from fixed_jtag_loader import validate_payload
from audit_canonical_multiseed import audit
from audit_board_bringup_evidence import receipt


def make(integration, destination):
    assert not destination.exists() and destination.is_relative_to(ROOT/'output')
    raw,_,residual,_=accepted_package();payload=validate_payload(integration)
    evidence={}
    def bind(p):
        p=Path(p);evidence[str(p)]=sha(p);return read_json(p)
    accepted_hashes=bind(integration/'accepted-inputs.json')
    assert accepted_hashes=={p.relative_to(ACCEPTED).as_posix():sha(p) for p in ACCEPTED.rglob('*') if p.is_file()}
    full=bind(integration/'rtl-full1/result.json');assert full['status']=='ACCEPTED_A_FULL_RTL_PASS'
    assert len(full['cases'])==6
    for r in full['cases']:
        case=integration/'rtl-full1'/r['case'];actual=bind(case/'result.json');assert actual==r
        meta=bind(case/'inputs.json');assert sha(case/'inputs.json')==r['inputs_sha256']
        assert sha(case/'simulation.log')==r['log_sha256']
        assert r['numerical_errors']==0 and r['complete_29_node_graph']
        assert r['scalar_checks']==65952000 and r['final_rgb_bytes']==2764800
        for file,digest in meta['files'].items():assert sha(file)==digest
        assert sha(meta['binary'])==meta['binary_sha256']
    short=bind(integration/'rtl-short1/result.json');assert short['status']=='ACCEPTED_A_SHORT_RTL_PASS'
    loader=receipt(integration/'jtag-loader2');assert loader['accepted_a'] and loader['status']=='PASS_FIXED_SOC_COMPLETE'
    bind(integration/'jtag-loader2/result.json')
    codec=receipt(integration/'jtag-host-codec1');assert any('PYTHON_HOST_CODEC' in s for s in codec['lines'])
    bind(integration/'jtag-host-codec1/result.json')
    testlog=integration/'host-regression5.log'
    log=testlog.read_text(encoding='utf-8-sig');assert 'Ran 11 tests' in log and log.rstrip().endswith('OK')
    evidence[str(testlog)]=sha(testlog)
    evidence[str(HERE/'fixed_jtag_loader.py')]=sha(HERE/'fixed_jtag_loader.py')
    evidence[str(HERE/'test_fixed_jtag_loader.py')]=sha(HERE/'test_fixed_jtag_loader.py')
    evidence[str(HERE/'accepted_a_assets.py')]=sha(HERE/'accepted_a_assets.py')
    evidence[str(HERE/'deploy_accepted_a.ps1')]=sha(HERE/'deploy_accepted_a.ps1')
    cdc=bind(integration/'cdc-review1.json');assert cdc['status']=='CDC_RESET_REVIEW_COMPLETE_FOR_BOUNDED_BRINGUP'
    for p,h in cdc['required_evidence'].items():assert sha(p)==h;evidence[p]=h
    # Verify original four routed reports; never re-run P&R for runtime data.
    timing=bind(ROOT/'output/toolchain/整板四seed原约束时序验收_2026-10-08.json')
    for r in timing['rows']:
        checked=audit(Path(r['canonical']));identity=checked.pop('identity')
        assert checked==r and identity==timing['identity'] and checked['current_timing_pass']
    board=Path(cdc['board']);bm=read_json(board/'inputs.json');active={Path(f).name:Path(f) for f in bm['source_files']}
    source=read_json(ROOT/'tmp/frame-execution/skip-credit-frame1/inputs.json')
    common=[]
    for r in source['inputs']:
        p=Path(r['snapshot'])
        if p.name in active:
            assert sha(p)==sha(active[p.name]),p.name
            common.append(p.name)
    assert len(common)>=40,common
    # Retain old NOT_RELEASED receipt; put the verified copy in this new bundle.
    bitroot=ROOT/'tmp/frame-execution/skip-credit-seed3-bitstream1'
    b=bind(bitroot/'result.json');g=bind(bitroot/'generation_inputs.json')
    assert sha(bitroot/'generation_inputs.json')==b['inputs_sha256']
    for name,h in g['input_geometry'].items():assert sha(bitroot/'outflow'/name)==h
    srcbit=bitroot/'outflow/ti60f225_oob.bit'
    bitsha=b['artifacts'][str(Path('outflow')/'ti60f225_oob.bit')]['sha256']
    assert sha(srcbit)==bitsha=='53aef0435a1ab3bef640631cd2710f45ab9057bd8862ec1c5a968032584b7ce7'
    destination.mkdir(parents=True)
    shutil.copy2(srcbit,destination/'ti60f225_oob.bit')
    for name,data in payload.items():(destination/(name+'.bin')).write_bytes(data)
    for name in ('schema.json','result.json'):
        shutil.copy2(integration/'descriptors'/name,destination/('program-'+name))
    shutil.copy2(HERE/'deploy_accepted_a.ps1',destination/'deploy.ps1')
    shutil.copy2(HERE/'accepted_a_bringup_README.md',destination/'README.md')
    plan=read_json(integration/'result.json')
    result=dict(status='RELEASED_FOR_BOUNDED_BRINGUP',package_sha256=PACKAGE_SHA,
        bitstream=str(destination/'ti60f225_oob.bit'),bitstream_sha256=bitsha,
        bundle_files={p.name:sha(p) for p in destination.iterdir() if p.is_file()},
        integration=str(integration),required_evidence=evidence,crc32=dict(coefficients=plan['padded_crc32'],**plan['descriptor_crc32']),
        residual_encoding=residual,shared_cnn_source_files=common,rtl_numeric=full['cases'],
        timing=timing['rows'],resources=timing['rows'][2]['resources'],
        permitted=['temporary JTAG configuration','idle identity/status inspection','4KiB bounded DDR readback/CRC',
                   'accepted A + compute/DMA upload with full readback','one-frame initial run after preceding hardware gates'],
        not_claimed=['physical USB/JTAG connection passed','DDR electrical/calibration/bandwidth passed on this physical board',
                     'sustained CSI-to-TMDS 30 AI fps','gesture/exit/evil-effect/FPS complete baseline','power-on standalone boot'],
        conditions=cdc['commissioning_conditions'],actual_board_test=False,
        accepted_directory_unchanged=True,
        usb_transport='USER1 direct fixed-controller protocol; no UART console; no RV32 injection; no Flash writes')
    write_json(destination/'release.json',result)
    print(result['status'],str(destination),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--integration',type=Path,default=ROOT/'tmp/frame-execution/accepted-a-integration1')
    p.add_argument('--destination',type=Path,default=ROOT/'output/deployment/B1-A-20261008');a=p.parse_args()
    make(a.integration.resolve(),a.destination.resolve())
