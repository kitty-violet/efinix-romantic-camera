"""Prepare an offline, supervised diagnostics bundle. No hardware access."""
import argparse,json,shutil
from pathlib import Path
from run_units import ROOT,HERE,sha
from audit_bringup_observation import audit as audit_observation
from audit_canonical_multiseed import audit as audit_canonical
from fixed_jtag_loader import validate_release,validate_payload


def prepare(board,batch,output):
    board,batch,output=board.resolve(),batch.resolve(),output.resolve()
    assert output.is_relative_to(ROOT/'output/deployment') and not output.exists()
    observation=audit_observation(board)
    terminal=json.loads((batch/'result.json').read_text())
    assert terminal['status']=='OFFLINE_PHYSICAL_EVIDENCE_COMPLETE_AWAIT_REVIEW'
    rows=[audit_canonical(Path(r['canonical'])) for r in terminal['seeds']]
    assert len(rows)==4 and len({r['seed'] for r in rows})==4
    assert all(r['current_timing_pass'] for r in rows)
    assert all(r['identity']==rows[0]['identity'] for r in rows)
    assert rows[0]['identity']['map']==observation['map_sha256']
    selected=terminal['selected'];canonical=Path(selected['canonical']);physical=Path(selected['physical'])
    chosen=next(r for r in rows if r['seed']==selected['seed'])
    assert chosen['physical']==str(physical) and chosen['canonical']==str(canonical)
    gray=Path(terminal['gray']);g=json.loads((gray/'result.json').read_text())
    gm=json.loads((gray/'inputs.json').read_text());gi=json.loads((gray/'clock_crossing_inventory.json').read_text())
    assert Path(gm['base'])==physical
    assert g['status']=='NEW_ROUTE_GRAY_SKEW_PASS_NOT_FULL_CDC_RELEASE'
    assert g['endpoints']==70 and g['max_skew_ns']<g['limit_ns']==2.5
    assert g['inputs_sha256']==sha(gray/'inputs.json')
    assert g['inventory_sha256']==sha(gray/'clock_crossing_inventory.json')
    assert gi['classes']['vendor_fifo_gray_first_stage']==70
    assert 1<=gi['classes']['read_only_diagnostic_first_stage']<=32
    for path,digest in gm['original'].items():assert sha(path)==digest
    generated=Path(terminal['bitstream']);br=json.loads((generated/'result.json').read_text())
    bm=json.loads((generated/'generation_inputs.json').read_text())
    assert br['status']=='OFFLINE_BITSTREAM_GENERATED_NOT_RELEASED' and br['inputs_sha256']==sha(generated/'generation_inputs.json')
    assert Path(bm['base'])==physical and Path(bm['canonical'])==canonical
    bit=generated/'outflow/ti60f225_oob.bit'
    bit_artifacts=[v for k,v in br['artifacts'].items()
                   if Path(k).as_posix()=='outflow/ti60f225_oob.bit']
    assert len(bit_artifacts)==1 and bit_artifacts[0]['sha256']==sha(bit)
    assert bit_artifacts[0]['bytes']==bit.stat().st_size
    parent=ROOT/'output/deployment/B1-A-20261008/release.json'
    old=validate_release(parent,validate_payload(ROOT/'tmp/frame-execution/accepted-a-integration1'))
    test=ROOT/'tmp/frame-execution/bringup-status-apb6'
    tr=json.loads((test/'result.json').read_text());tm=json.loads((test/'inputs.json').read_text())
    assert tr['status']=='BRINGUP_STATUS_APB_FOURSTATE_PASS' and tr['inputs_sha256']==sha(test/'inputs.json')
    # Subsequent top-only synthesis attributes do not alter the tested APB.
    apb=next(r for r in tm['inputs'] if Path(r['snapshot']).name=='cnn_apb_control.sv')
    assert sha(board/'assembly/cnn_apb_control.sv')==sha(apb['snapshot'])==apb['sha256']
    sync=ROOT/'tmp/frame-execution/bringup-sync-map2';sr=json.loads((sync/'result.json').read_text())
    assert sr['status']=='BRINGUP_SYNC_64_FF_NO_SRL_PASS'
    assert sr['top_sha256']==sha(board/'rtl/ti60f225_oob_top.v')
    hostlog=batch/'host-regression3.log'
    assert 'Ran 7 tests' in hostlog.read_text() and hostlog.read_text().rstrip().endswith('OK')
    evidence=dict(old['required_evidence'])
    for path in (parent,board/'inputs.json',board/'result.json',board/'observation_audit.json',
                 batch/'result.json',Path(terminal['multiseed_receipt']),gray/'result.json',gray/'inputs.json',
                 gray/'clock_crossing_inventory.json',gray/'gray_skew_audit.json',
                 generated/'result.json',generated/'generation_inputs.json',test/'result.json',test/'inputs.json',
                 test/'simulation.log',sync/'result.json',sync/'inputs.json',hostlog,
                 HERE/'read_bringup_status.py',HERE/'test_read_bringup_status.py',Path(__file__)):
        evidence[str(path)]=sha(path)
    for row in rows:
        for path in (Path(row['canonical'])/'result.json',Path(row['canonical'])/'inputs.json',
                     Path(row['physical'])/'physical_inputs.json'):
            evidence[str(path)]=sha(path)
    for row in json.loads((board/'inputs.json').read_text())['inputs']:
        evidence[row['snapshot']]=row['sha256']
    output.mkdir(parents=True)
    shutil.copy2(bit,output/'ti60f225_oob.bit')
    (output/'README.md').write_text('''# 只读上板诊断准备包

**等待用户回来监督。当前不得自动下载或访问实物。**

此包从原冻结主视频系统只增加状态观察；没有改复位、计算、协议、引脚、频率或PHY。
离线门禁与位流SHA见manifest.json；不代表DDR、画面或实板性能已经通过。

用户明确恢复后：核验清单及当前USB B接口 → 官方JTAG临时配置本bit → 只读身份和3组状态。
不写Flash、不发DDR请求、不装载A、不启动CNN。依状态定位首因后再决定下一项受限测试。
不得将本包冒充原A部署release.json，也不能沿用旧Memtest/Load收据。

只读工具为项目hw/efinity/prototypes/frame_execution/read_bringup_status.py，
使用本次枚举的显式物理B URL以及一个新的report路径；宿主总上限20秒。
读失败保留有效前缀并停止，不盲重试，不按CRESET_N加载未知Flash。
''',encoding='utf-8')
    result=dict(status='OFFLINE_READY_AWAIT_USER_SUPERVISION',board=str(board),batch=str(batch),
                selected_seed=selected['seed'],bitstream=str(output/'ti60f225_oob.bit'),
                bitstream_sha256=sha(bit),bitstream_bytes=bit.stat().st_size,
                resources=chosen['resources'],wns_ns=chosen['wns_ns'],whs_ns=chosen['whs_ns'],
                cdc_classes=gi['classes'],gray_max_skew_ns=g['max_skew_ns'],
                required_evidence=evidence,user_supervision_required=True,hardware_access=False,
                automatic_download_allowed=False,flash_write_allowed=False,ddr_test_released=False,
                a_weight_load_released=False,full_system_accepted=False,
                bundle_files={p.name:sha(p) for p in output.iterdir() if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for path,digest in evidence.items():assert sha(path)==digest
    print(json.dumps({k:result[k] for k in ('status','bitstream','bitstream_sha256','selected_seed','resources','wns_ns','whs_ns')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--board',required=True,type=Path)
    p.add_argument('--batch',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();prepare(a.board,a.batch,a.output)
