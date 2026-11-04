"""Review the new partial-RAM placement without altering prior evidence."""
import runpy
from run_units import ROOT,HERE,sha
from control_command_repair import once


def main():
    source=HERE/'review_snapshot_ram_route.py';code=source.read_text()
    code=once(code,"PHYSICAL=ROOT/'tmp/frame-execution/snapshot-ram-place1'",
              "PHYSICAL=ROOT/'tmp/frame-execution/snapshot-arena-free-place1'")
    code=once(code," for ref in ('split-registered-place1','split-outputs-place3','startup-cause-place2'):",
              " for ref in ('snapshot-ram-place1','split-registered-place1','split-outputs-place3','startup-cause-place2'):")
    code=once(code,"for n in ('ti60f225_oob.sdc','fixed_dsp.placeloc','ti60f225_oob.peri.xml'):",
              "for n in ('ti60f225_oob.sdc','ti60f225_oob.peri.xml'):")
    anchor=" comparison={};paths=[PHYSICAL/x for x in"
    additional=""" from partial_cnn_ram_locations import verify_placed
 from fixed_control_ddr_reuse_locations import locations
 measured=json.loads((PHYSICAL/'cnn_ram_actual_locations.json').read_text())
 assert measured==json.loads(json.dumps(verify_placed(PHYSICAL)))
 origin=ROOT/'tmp/frame-execution/snapshot-ram-place1'
 assert sha(PHYSICAL/'outflow/ti60f225_oob.map.v')==sha(origin/'outflow/ti60f225_oob.map.v')
 for suffix in ('map.v','vdb','raminfo.pb'):
  name='outflow/ti60f225_oob.'+suffix
  assert sha(PHYSICAL/name)==sha(BOARD/name)==sha(origin/name),(name,'Synthesis payload mismatch')
 before=locations(origin/'fixed_dsp.placeloc');after=locations(PHYSICAL/'fixed_dsp.placeloc')
 removed=set(before)-set(after)
 assert len(removed)==84 and all(n.startswith('video_processing_pipeline/pipeline/cnn/core/arena/') for n in removed)
 assert after=={n:p for n,p in before.items() if n not in removed}
 assert len(after)==317 and measured['retained']==29 and measured['released']==84
"""
    code=once(code,anchor,additional+anchor)
    code=once(code," paths += [EVIDENCE/'result.json'",
              " paths += [PHYSICAL/'cnn_ram_placement.json',PHYSICAL/'fixed_dsp.placeloc',PHYSICAL/'outflow/ti60f225_oob.vdb',BOARD/'outflow/ti60f225_oob.vdb',PHYSICAL/'outflow/ti60f225_oob.raminfo.pb',HERE/'partial_cnn_ram_locations.py',HERE/'place_snapshot_arena_free.py',Path(__file__),EVIDENCE/'result.json'")
    # Binding the generated wrapper and its source is separate from the
    # hardware evidence. The original reviewer remains unchanged.
    driver=ROOT/'tmp/frame-execution/snapshot-arena-free-review-driver.py'
    compile(code,str(driver),'exec')
    if driver.exists():assert driver.read_text()==code
    else:driver.write_text(code,encoding='utf8')
    import json
    metadata=dict(source=str(source),source_sha256=sha(source),generated_sha256=sha(driver),
                  wrapper_sha256=sha(__file__))
    record=driver.with_suffix('.json')
    if record.exists():assert json.loads(record.read_text())==metadata
    else:record.write_text(json.dumps(metadata,indent=2))
    runpy.run_path(str(driver),run_name='__main__')


if __name__=='__main__':main()
