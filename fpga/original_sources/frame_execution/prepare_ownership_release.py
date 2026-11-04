"""Assemble the complete 150MHz video/A-CNN system; initial access is staged."""
import argparse,json,shutil
from pathlib import Path
from run_units import ROOT,HERE,sha
from registered_ownership_board import audit_delta
from ownership_evidence import bind_proofs
from audit_canonical_multiseed import audit as physical_audit
from audit_measured_pinout import audit as pin_audit


def prepare(board,batch,gray,bitroot,review,output,selection_audit=None):
    assert output.is_relative_to(ROOT/'output/deployment') and not output.exists()
    delta=audit_delta(board)
    parent=ROOT/'output/deployment/board-local-reset-20261009/manifest.json'
    previous=json.loads(parent.read_text());evidence=dict(previous['required_evidence'])
    for path,digest in evidence.items():assert sha(path)==digest,('Parent evidence changed',path)
    for name,digest in previous['bundle_files'].items():assert sha(parent.parent/name)==digest
    evidence[str(parent)]=sha(parent)
    def bind(path):
        evidence[str(path)]=sha(path)
        return json.loads(path.read_text())
    selected_review=None
    if selection_audit:
        from audit_ownership_selection import selection
        selected_review=bind(selection_audit)
        assert selected_review==selection(board,batch,Path(selected_review['selected']['canonical']))
        evidence.update(selected_review['required_evidence'])
        rows=[selected_review['selected']]
    else:
        terminal=bind(batch/'result.json')
        assert terminal['status']=='REGISTERED_OWNERSHIP_FOUR_SEED_PASS_PENDING_CDC'
        multiseed=bind(Path(terminal['evidence']))
        assert sha(terminal['evidence'])==terminal['evidence_sha256']
        rows=[physical_audit(Path(r['canonical'])) for r in multiseed['rows']]
        assert len(rows)==4 and len({r['seed'] for r in rows})==4
        assert multiseed['identity']==rows[0]['identity']
        assert multiseed['status']=='FOUR_SEED_PHYSICAL_BASIC_TIMING_PASS_NOT_BOARD_RELEASE'
    functional=bind_proofs(board,evidence)
    assert all(r['current_timing_pass'] and r['identity']==rows[0]['identity'] for r in rows)
    mapped=bind(board/'result.json');assert rows[0]['identity']['map']==mapped['map_sha256']
    assert mapped['inputs_sha256']==sha(board/'inputs.json')
    gm=bind(gray/'inputs.json')
    chosen=next(r for r in rows if Path(r['physical'])==Path(gm['base']))
    physical=Path(chosen['physical']);canonical=Path(chosen['canonical'])
    reset=bind(batch/'implementation.json')
    assert reset['status']=='CONTROL_STATE_MAPPED_RESET_AND_PERIPHERY_PASS'
    assert reset['board']==str(board) and reset['reset_ff']==18 and reset['observation_ff']==64
    for path,digest in reset['required_evidence'].items():assert sha(path)==digest
    evidence.update(reset['required_evidence'])
    cdc=bind(review)
    assert cdc['status']=='REGISTERED_OWNERSHIP_CDC_REVIEW_FOR_SUPERVISED_STARTUP'
    assert cdc['board']==str(board) and cdc['gray']==str(gray)
    assert cdc['unknown_new_crossings']==0 and cdc['ddr_workload_released'] is False
    for path,digest in cdc['required_evidence'].items():assert sha(path)==digest
    evidence.update(cdc['required_evidence'])
    g=bind(gray/'result.json');gm=bind(gray/'inputs.json');gi=bind(gray/'clock_crossing_inventory.json')
    assert g['status']=='NEW_ROUTE_GRAY_SKEW_PASS_NOT_FULL_CDC_RELEASE'
    assert Path(gm['base'])==physical and g['buses']==10 and g['endpoints']==70
    assert g['max_skew_ns']<g['limit_ns']==2.5
    assert g['inputs_sha256']==sha(gray/'inputs.json') and g['inventory_sha256']==sha(gray/'clock_crossing_inventory.json')
    for p,h in gm['original'].items():assert sha(p)==h
    br=bind(bitroot/'result.json');bm=bind(bitroot/'generation_inputs.json')
    assert br['status']=='OFFLINE_BITSTREAM_GENERATED_NOT_RELEASED'
    assert br['inputs_sha256']==sha(bitroot/'generation_inputs.json')
    assert Path(bm['base'])==physical and Path(bm['canonical'])==canonical
    bit=bitroot/'outflow/ti60f225_oob.bit'
    artifact=next(v for k,v in br['artifacts'].items() if Path(k).name=='ti60f225_oob.bit')
    assert sha(bit)==artifact['sha256'] and bit.stat().st_size==artifact['bytes']
    pins=[]
    for row in rows:
        pa=pin_audit(Path(row['physical']));pins.append(pa);evidence.update(pa['inputs'])
        routed=Path(row['physical'])
        if (routed/'cnn_ram_placement.json').exists():
            from fixed_cnn_ram_locations import verify_placed
            actual=verify_placed(routed)
            assert bind(routed/'cnn_ram_actual_locations.json')==actual
            ram=bind(routed/'cnn_ram_placement.json')
            for item in (routed/'fixed_dsp.placeloc',routed/'efx_pnr_settings.ini',
                         routed/'cnn_ram_generator.py',routed/'physical_runner.py',
                         Path(ram['reference_canonical'])/'result.json',
                         Path(ram['reference_physical'])/'outflow/ti60f225_oob.out.placeloc',
                         HERE/'fixed_cnn_ram_locations.py',HERE/'probe_control_state_ram_physical.py'):
                evidence[str(item)]=sha(item)
        for p in (Path(row['canonical'])/'result.json',Path(row['canonical'])/'inputs.json',Path(row['physical'])/'physical_inputs.json'):
            bind(p)
    for row in bind(board/'inputs.json')['inputs']:evidence[row['snapshot']]=row['sha256']
    for path in (batch/'inputs.json',batch/'stages.json',gray/'gray_skew_audit.json',
                 batch/'host_tests.log',HERE/'test_control_snapshot_host.py',
                 HERE/'read_control_snapshot.py',ROOT/'tmp/board-test/program-bringup-status.py',Path(__file__)):
        evidence[str(path)]=sha(path)
    hostlog=(batch/'host_tests.log').read_text()
    assert 'Ran 11 tests' in hostlog and hostlog.rstrip().endswith('OK')
    output.mkdir(parents=True);shutil.copy2(bit,output/'ti60f225_oob.bit')
    # Stage the accepted assets, but do not issue any hardware transaction.
    accepted=ROOT/'model-training-workspace/accepted/B1-A-20261008'
    descriptors=ROOT/'tmp/frame-execution/accepted-a-integration1/descriptors'
    for f,name in ((accepted/'coefficients-padded.bin','coefficients-padded.bin'),
                   (descriptors/'compute.bin','compute.bin'),(descriptors/'dma.bin','dma.bin')):
        shutil.copy2(f,output/name)
    (output/'README.md').write_text('''# 完整视频与A风格链：150MHz监督上板候选

硬件保留摄像头、DDR、A风格CNN、显示及HDMI，内部计算保持150MHz。
本轮为缩短组合路径，寄存RAM访问归属及RQ任务选择；新增等待周期。
真实A单张720p完整CNN数值、四态受影响模块、并发回压/取消/恢复及所选实现原约束证据见manifest.json。
采用已冻结且通过的seed；其他seed失败如实保留，不用于替代所选实现的独立检查。
CNN行为DDR测量不是摄像头到屏幕实测；新增手势等尚未完成的基线职责仍需后续接入。

先在用户监督下临时JTAG配置、读取selectors2/3状态；至多一次有超时的APB身份探测。
这是完整系统的首个硬件检查步骤，没有为诊断裁掉主视频链。
原板启动/总线应答与DDR校准尚待实物确认，通过后才解锁4KiB受限CRC、A装载和单帧自动停止。
本包已附A权重及同步生成的compute/DMA程序，不能绕过硬件检查直接运行。
禁止Flash写入。USB按物理B接口重新枚举，不依赖固定COM号。
program.py默认仅离线核对全部证据及文件哈希，不连接板子。
''',encoding='utf-8')
    manifest=dict(status='OFFLINE_READY_AWAIT_USER_SUPERVISION',board=str(board),batch=str(batch),
        selected_seed=chosen['seed'],bitstream=str(output/'ti60f225_oob.bit'),bitstream_sha256=sha(bit),
        bitstream_bytes=bit.stat().st_size,resources=chosen['resources'],wns_ns=chosen['wns_ns'],whs_ns=chosen['whs_ns'],
        observation_delta=delta,functional=functional,full_video_hardware_preserved=True,compute_clock_mhz=150,cdc_classes=gi['classes'],gray_max_skew_ns=g['max_skew_ns'],generated_pin_audits=pins,
        required_evidence=evidence,selection_review=selected_review,user_supervision_required=True,hardware_access=False,
        automatic_download_allowed=False,flash_write_allowed=False,ddr_test_released=False,
        a_weight_load_released=False,full_system_accepted=False,
        bundle_files={name:sha(output/name) for name in ('README.md','ti60f225_oob.bit','coefficients-padded.bin','compute.bin','dma.bin')})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for path,digest in evidence.items():assert sha(path)==digest,('Evidence changed',path)
    # Frozen wrapper delegates to the already exercised official procedure.
    wrapper='''import importlib.util,json
from pathlib import Path
BASE=Path('E:/fpga/tmp/board-test')
spec=importlib.util.spec_from_file_location('bounded_programmer',BASE/'program-bringup-status.py')
programmer=importlib.util.module_from_spec(spec);spec.loader.exec_module(programmer)
programmer.BUNDLE=Path(__file__).resolve().parent
programmer.MANIFEST_SHA=@@MANIFEST@@
programmer.BIT_SHA=@@BIT@@
def validate():
 b=programmer.BUNDLE;m=b/'manifest.json'
 assert programmer.sha(m)==programmer.MANIFEST_SHA,'Frozen manifest mismatch'
 d=json.loads(m.read_text());bit=b/'ti60f225_oob.bit'
 assert d['status']=='OFFLINE_READY_AWAIT_USER_SUPERVISION'
 assert Path(d['bitstream'])==bit and programmer.sha(bit)==programmer.BIT_SHA==d['bitstream_sha256']
 assert bit.stat().st_size==d['bitstream_bytes'] and d['wns_ns']>=0 and d['whs_ns']>=0
 assert d['user_supervision_required'] is True
 for flag in ('automatic_download_allowed','flash_write_allowed','ddr_test_released','a_weight_load_released','full_system_accepted'):
  assert d[flag] is False
 for path,h in d['required_evidence'].items():assert programmer.sha(path)==h,('Evidence mismatch',path)
 for name,h in d['bundle_files'].items():assert programmer.sha(b/name)==h,('Bundle mismatch',name)
 return dict(manifest_sha256=programmer.MANIFEST_SHA,bitstream_sha256=programmer.BIT_SHA,bitstream=str(bit),
             evidence_files=len(d['required_evidence']),host_script_sha256=programmer.sha(__file__))
programmer.validate=validate
if __name__=='__main__':programmer.main()
'''.replace('@@MANIFEST@@',repr(sha(output/'manifest.json'))).replace('@@BIT@@',repr(sha(bit)))
    compile(wrapper,str(output/'program.py'),'exec');(output/'program.py').write_text(wrapper,encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ('status','selected_seed','bitstream_sha256','resources','wns_ns','whs_ns')}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('board','batch','gray','bitroot','review','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--selection-audit',type=Path)
    a=p.parse_args();prepare(**{k:v.resolve() if v is not None else None for k,v in vars(a).items()})
