"""Current candidate: all retained Gray/camera/JTAG checks plus service crossings."""
import argparse,json,runpy,sys
from pathlib import Path
from run_units import HERE,ROOT,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--inventory',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();base=a.base.resolve();struct=a.inventory.resolve()
    assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    source=HERE/'audit_snapshot_routed_gray.py';code=source.read_text()
    info=json.loads(struct.read_text())
    compact=bool(info.get('dualclock_sample_ram')) or any(r['start'].startswith('service_serializer/held[') for r in info['cross_domain_cones'])
    service_checker='audit_video_service_compact_routed_cdc' if compact else 'audit_video_service_routed_cdc'
    if compact:
        needle='from audit_snapshot_jtag_paths import audit as audit_jtag';assert code.count(needle)==1
        code=code.replace(needle,'from audit_video_service_compact_jtag import audit as audit_jtag')
    replacements=[
      ('from windows_process_tree import stop_launched_tree','from windows_process_tree_safe import stop_launched_tree'),
      ('from snapshot_clock_cdc import inventory','from camera_joint_cdc import inventory\nfrom '+service_checker+' import inventory as service_inventory'),
      ("here/'snapshot_clock_cdc.py'","here/'camera_joint_cdc.py'"),
      ("canonical=b/'canonical_timing.sdc'\nassert sha(canonical)==meta['setup_margin']['canonical_sdc_sha256']",
       "canonical=b/'ti60f225_oob.sdc'\nassert meta['setup_margin'] is None\nassert sha(canonical)==next(x['sha256'] for x in meta['inputs'] if Path(x['snapshot'])==canonical)"),
      ("here=Path(__file__).parent","here=Path("+repr(str(HERE))+")"),
      ("assert sum(line.startswith('set_bus_skew ') for line in uncut.splitlines())==10",
       "video_group='set_clock_groups -asynchronous -group {vid_clk_dvi2} -group {process_clk_150} -group {hdmi_tx_slow_clk}'\nassert uncut.count(video_group)==1\nuncut=uncut.replace(video_group,'# Diagnostic only: expose pixel/system held mailbox; original acceptance unchanged.')\nassert sum(line.startswith('set_bus_skew ') for line in uncut.splitlines())==10"),
      (" ('system_to_jtag','process_clk_150','jtag_inst1_TCK'),",
       " ('camera_cfg_to_system','CLK_5M','process_clk_150'),\n ('camera_pixel_to_system','i_mipi_rx_pclk','process_clk_150'),\n ('service_pixel_to_system','hdmi_tx_slow_clk','process_clk_150'),\n ('service_system_to_pixel','process_clk_150','hdmi_tx_slow_clk'),\n ('system_to_jtag','process_clk_150','jtag_inst1_TCK'),"),
      ('    inv=inventory(r)',
       "    service=service_inventory(r,Path("+repr(str(struct))+"))\n    (r/'service_cdc.json').write_text(json.dumps(service,indent=2))\n    inv=inventory(r)"),
      ("inventory_sha256=sha(r/'clock_crossing_inventory.json'),",
       "inventory_sha256=sha(r/'clock_crossing_inventory.json'),service_cdc_sha256=sha(r/'service_cdc.json'),"),
    ]
    for old,new in replacements:
        assert code.count(old)==1,old;code=code.replace(old,new)
    driver=root.parent/(root.name+'-service-driver.py');assert not driver.exists()
    compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
        wrapper_sha256=sha(__file__),driver_sha256=sha(driver),structure_sha256=sha(struct),
        extra_checker_sha256=sha(HERE/(service_checker+'.py')),
        compact_jtag_checker_sha256=sha(HERE/'audit_video_service_compact_jtag.py') if compact else None,
        original_constraints_unchanged=True),indent=2))
    sys.argv=[str(driver),'--base',str(base),'--root',str(root)]
    runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
