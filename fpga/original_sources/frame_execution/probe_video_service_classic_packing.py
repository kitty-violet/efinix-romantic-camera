"""Bounded capacity comparison of the documented classic packing flow.

No logic, placement anchors, clocks, pins, seed or acceptance threshold change.
Stops after packing; this experiment does not start another long route.
"""
import argparse, json, runpy, subprocess, sys
from pathlib import Path
from run_units import HERE, ROOT, EF, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--board', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    assert root.is_relative_to(ROOT/'tmp/frame-execution') and not root.exists()
    docs = EF/'doc/topics/tim-pnr.html'
    assert '--classic_flow' in docs.read_text(encoding='utf8')
    subprocess.run([sys.executable, str(HERE/'place_video_service_epoch.py'),
                    '--board', str(a.board.resolve()), '--root', str(root), '--prepare-only'],
                   check=True, timeout=120)
    driver = root.parent/(root.name+'-wrapper.py')
    receipt = driver.with_suffix('.json')
    meta = json.loads(receipt.read_text())
    assert sha(driver) == meta['driver_sha256']
    code = driver.read_text()
    anchor = "    code=once(code,'related_gray_layout=None\\n',"
    change = '''xml=root/'ti60f225_oob.xml';before_classic=xml.read_text()
classic_tag='<efx:param name="classic_flow" value="off" value_type="e_noarg" />'
assert before_classic.count(classic_tag)==1
xml.write_text(before_classic.replace(classic_tag,classic_tag.replace('value="off"','value="on"')),encoding='utf8')
for r in records:
 if Path(r['snapshot'])==xml:r['sha256']=sha(xml)
(root/'classic_flow_change.json').write_text(json.dumps(dict(option='classic_flow',before='off',after='on',clocks_pins_rtl_unchanged=True),indent=2))
seed_change=None'''
    assert code.count(anchor) == 1
    code = code.replace(anchor, "    code=once(code,'seed_change=None',"+repr(change)+")\n"+anchor)
    compile(code, str(driver), 'exec')
    driver.write_text(code, encoding='utf8')
    meta['argv'].append('--packing-only')
    meta.update(driver_sha256=sha(driver), capacity_adapter_sha256=sha(__file__),
                documented_option_source=str(docs), documented_option_sha256=sha(docs),
                change='Classic packing flow capacity comparison only; current route untouched.')
    receipt.write_text(json.dumps(meta, indent=2))
    sys.argv = meta['argv']
    runpy.run_path(str(driver), run_name='__main__')


if __name__ == '__main__':
    main()
