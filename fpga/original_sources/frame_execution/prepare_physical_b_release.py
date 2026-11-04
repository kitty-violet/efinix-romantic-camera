"""Derive a host-binding-only release while verifying and retaining its parent."""
from pathlib import Path
import shutil
from accepted_a_assets import ROOT, HERE, read_json, write_json, sha
from fixed_jtag_loader import validate_payload, validate_release


def main():
    old = ROOT/'output/deployment/B1-A-20261008'
    new = ROOT/'output/deployment/B1-A-20261008-usb1'
    payload = validate_payload(ROOT/'tmp/frame-execution/accepted-a-integration1')
    parent = validate_release(old/'release.json', payload)
    regression = ROOT/'tmp/board-test/20261008-physical-b-host-regression1.log'
    text = regression.read_text(encoding='utf-8-sig')
    assert 'Ran 19 tests' in text and text.rstrip().endswith('OK')
    assert not new.exists()
    new.mkdir()
    for name in parent['bundle_files']:
        if name not in ('README.md', 'deploy.ps1'):
            shutil.copy2(old/name, new/name)
    launcher = (old/'deploy.ps1').read_text(encoding='utf-8-sig')
    launcher = launcher.replace("[string]$Bundle='E:\\fpga\\output\\deployment\\B1-A-20261008'", "[string]$Bundle='E:\\fpga\\output\\deployment\\B1-A-20261008-usb1'")
    launcher = launcher.replace('frame_execution\\fixed_jtag_loader.py', 'frame_execution\\fixed_jtag_physical_b.py')
    launcher = launcher.replace('Use the enumerated FT4232 serial URL, JTAG interface /2.', 'Use the current physical-B USB URL; virtual B uses /1, full device uses /2. Descriptor identity is checked.')
    assert 'fixed_jtag_physical_b.py' in launcher
    (new/'deploy.ps1').write_text(launcher, encoding='utf-8')
    (new/'README.md').write_text('''# 定稿A受限板测：独立B接口适配

本副本仅修正主机USB接口识别，原位流、A权重、计算/DMA程序及全部RTL保持不变。
父包 B1-A-20261008 与原证据不覆盖。新主机按 0403:6011、物理接口1和端点83/04核对B，拒绝A/C/D。
独立驱动把物理B呈现为单接口虚拟设备，因此URL尾部为/1；整体四接口设备仍选/2。
URL总线/地址以本次实际枚举为准，插拔后重新核对，不能固定COM号。

入口：本目录 deploy.ps1，动作 Validate、Devices、Inspect、Memtest、Load、Run。
参数与父包相同：-Url 当前B地址 -Report 新记录路径；Load/Run需要-PriorReport。
仍先临时JTAG配置，再Inspect、4KiB Memtest、全量Load回读；输入已有效才Run一帧。
不写Flash。量化、帧计数、硬件超时、地址限制和其余放行边界与父包相同。
19项离线回归通过；真实存储、装载和运行结果需另看板测收据，不能从主机回归推断。
''', encoding='utf-8')
    result = dict(parent)
    result.update(bitstream=str(new/'ti60f225_oob.bit'),
                  parent_release=str(old/'release.json'), parent_release_sha256=sha(old/'release.json'),
                  host_binding_extension='physical B verified by descriptor; no FPGA/netlist change',
                  bundle_files={p.name:sha(p) for p in new.iterdir() if p.is_file()})
    result['required_evidence'] = dict(parent['required_evidence'])
    for path in (old/'release.json', HERE/'fixed_jtag_physical_b.py', HERE/'test_fixed_jtag_physical_b.py', Path(__file__), regression):
        result['required_evidence'][str(path)] = sha(path)
    write_json(new/'release.json', result)
    validate_release(old/'release.json', payload)
    validate_release(new/'release.json', payload)
    print('PARENT_AND_USB_EXTENSION_VALID', new, sha(new/'release.json'))


if __name__ == '__main__': main()
