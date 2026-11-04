import json, shutil, tempfile, unittest
from pathlib import Path
from board_measured_pinmap import BASE, VARIANT, ROOT, pin_audit, audit
from measured_pinmap_clock_profile import apply_profile

class PinmapGate(unittest.TestCase):
    def test_exact_gpio_correction(self):
        self.assertEqual(len(pin_audit(VARIANT)),5)
    def test_old_mapping_rejected(self):
        with self.assertRaises(AssertionError):pin_audit(BASE/'ti60f225_oob.peri.xml')
    def test_key_output_and_clock_change_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as d:
            p=Path(d)/'bad.peri.xml';body=VARIANT.read_text()
            for wrong in [body.replace('name="i_arstn" gpio_def="GPIOR_22" mode="input"',
                                     'name="i_arstn" gpio_def="GPIOR_22" mode="output"'),
                          body.replace('ref_clock_freq="25.0000"','ref_clock_freq="26.0000"')]:
                p.write_text(wrong)
                with self.assertRaises(AssertionError):pin_audit(p)
    def test_unchanged_map_and_clock_constraints(self):
        b=ROOT/'tmp/frame-execution/connected-board-measured-pinmap1'
        self.assertEqual(audit(b)['status'],'MEASURED_GPIO_DELTA_SAME_RTL_MAP_PASS')
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as d:
            p=Path(d)
            for n in ('ti60f225_oob.peri.xml','ti60f225_oob.sdc','ddr516_reference.sv','ddr516_v4_controller.sv'):
                shutil.copy2(b/n,p/n)
            profile=apply_profile(p,ROOT/'tmp/frame-execution/axi-valid-payload-physical1/clock_crossing_inventory.json')
            self.assertEqual(profile['gray_bus_constraints'],10)
            previous=ROOT/'tmp/frame-execution/bringup-status2-physical-batch1/pnr-seed1/canonical_timing.sdc'
            self.assertEqual((p/'ti60f225_oob.sdc').read_bytes(),previous.read_bytes())

if __name__=='__main__':unittest.main(verbosity=2)
