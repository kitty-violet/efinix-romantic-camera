"""Assemble a frozen read-only diagnostic bundle after all offline gates."""
import argparse,json,shutil
from pathlib import Path
from run_units import ROOT,HERE,sha
from prepare_snapshot_board import audit_delta
from audit_canonical_multiseed import audit as physical_audit
from audit_measured_pinout import audit as pin_audit


def prepare(board,batch,gray,bitroot,review,output):
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
    terminal=bind(batch/'result.json')
    assert terminal['status']=='SNAPSHOT_FOUR_SEED_BASIC_TIMING_PASS_PENDING_CDC'
    rows=[physical_audit(Path(r['canonical'])) for r in terminal['seeds']]
    assert len(rows)==4 and len({r['seed'] for r in rows})==4
    assert all(r['current_timing_pass'] and r['identity']==rows[0]['identity'] for r in rows)
    mapped=bind(board/'result.json');assert rows[0]['identity']['map']==mapped['map_sha256']
    assert mapped['inputs_sha256']==sha(board/'inputs.json')
    chosen=next(r for r in rows if r['seed']==terminal['selected']['seed'])
    physical=Path(chosen['physical']);canonical=Path(chosen['canonical'])
    multiseed=bind(Path(terminal['multiseed_receipt']))
    assert multiseed['status']=='FOUR_SEED_PHYSICAL_BASIC_TIMING_PASS_NOT_BOARD_RELEASE'
    reset=bind(batch/'snapshot_implementation.json')
    assert reset['status']=='FULL_MAPPED_SNAPSHOT_RESET_AND_PERIPHERY_PASS'
    assert reset['board']==str(board) and reset['reset_ff']==18 and reset['observation_ff']==64
    for path,digest in reset['required_evidence'].items():assert sha(path)==digest
    evidence.update(reset['required_evidence'])
    cdc=bind(review)
    assert cdc['status']=='SNAPSHOT_CDC_REVIEW_FOR_READ_ONLY_DIAGNOSTICS'
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
        for p in (Path(row['canonical'])/'result.json',Path(row['canonical'])/'inputs.json',Path(row['physical'])/'physical_inputs.json'):
            bind(p)
    for row in bind(board/'inputs.json')['inputs']:evidence[row['snapshot']]=row['sha256']
    for path in (batch/'inputs.json',batch/'all_attempts.json',gray/'gray_skew_audit.json',
                 batch/'host_tests.log',HERE/'test_control_snapshot_host.py',
                 HERE/'read_control_snapshot.py',ROOT/'tmp/board-test/program-bringup-status.py',Path(__file__)):
        evidence[str(path)]=sha(path)
    hostlog=(batch/'host_tests.log').read_text()
    assert 'Ran 11 tests' in hostlog and hostlog.rstrip().endswith('OK')
    output.mkdir(parents=True);shutil.copy2(bit,output/'ti60f225_oob.bit')
    (output/'README.md').write_text('''# 独立控制观察：监督只读诊断包

用途：通过独立JTAG selectors 2/3读取时钟、复位、命令接收/接受/返回计数、APB状态。
定位现有f8100000身份读取超时。保留原视频、CNN、DDR、时钟、引脚与功能复位。
普通selector0/1协议保留。新观察器采用20个实际同步FF和稳定保持总线。

用户在旁监督时才允许临时JTAG配置。禁止Flash、DDR负载、A装载或CNN Run。
重新枚举物理B接口，不沿用旧URL或COM号；program.py默认只做离线校验。
配置后使用read_control_snapshot.py先只读观察，再选择至多一次--probe-identity。
宿主20秒总超时，失败保留独立状态；不要盲目重复读请求，也不要按CRESET_N。

离线数值/接口短测、四seed原约束、实际同步FF/Gray和握手路径证据见manifest.json。
本包允许受监督诊断，不代表系统寄存器、DDR校准、画面或实际吞吐已经通过。
''',encoding='utf-8')
    manifest=dict(status='OFFLINE_READY_AWAIT_USER_SUPERVISION',board=str(board),batch=str(batch),
        selected_seed=chosen['seed'],bitstream=str(output/'ti60f225_oob.bit'),bitstream_sha256=sha(bit),
        bitstream_bytes=bit.stat().st_size,resources=chosen['resources'],wns_ns=chosen['wns_ns'],whs_ns=chosen['whs_ns'],
        observation_delta=delta,cdc_classes=gi['classes'],gray_max_skew_ns=g['max_skew_ns'],generated_pin_audits=pins,
        required_evidence=evidence,user_supervision_required=True,hardware_access=False,
        automatic_download_allowed=False,flash_write_allowed=False,ddr_test_released=False,
        a_weight_load_released=False,full_system_accepted=False,
        bundle_files={'README.md':sha(output/'README.md'),'ti60f225_oob.bit':sha(bit)})
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
    a=p.parse_args();prepare(**{k:v.resolve() for k,v in vars(a).items()})


