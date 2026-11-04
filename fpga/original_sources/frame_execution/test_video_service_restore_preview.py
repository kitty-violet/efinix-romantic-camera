"""Recovery regression: shared ISP mailbox, invalidated READY, failure gating."""
import copy
import json
import sys
import unittest
from pathlib import Path
import video_service_restore_preview as restore

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.plan = restore.commands(Path('manifest'), Path('receipt'), Path('out'))
        base = restore.ROOT/'tmp/board-test/video-service69-board1'
        self.rows = {n: json.loads((base/f'scene-restore-{file}1.json').read_text())
                     for n, file in [('clear', 'clear'), ('prepare', 'prepare'), ('live', 'live')]}
        self.calls = []
    def run_plan(self, fail=None):
        def execute(name, command):
            self.calls.append(name)
            return 1 if name == fail else 0
        return restore.sequence(self.plan, execute, self.rows.__getitem__, lambda *_: None)
    def test_recorded_recovery(self):
        self.assertIn('NEW_COMMIT', self.run_plan())
        self.assertEqual(self.calls, ['clear', 'prepare', 'live'])
    def test_isp_low_bit_never_selects_pattern(self):
        command = self.plan[0][1]
        self.assertEqual(command[command.index('--tx-pattern')+1], 'off')
        self.assertNotIn('--mode', command)
    def test_failure_never_starts(self):
        for stage, expected in [('clear', ['clear']), ('prepare', ['clear', 'prepare'])]:
            self.calls = []
            with self.assertRaises(RuntimeError): self.run_plan(stage)
            self.assertEqual(self.calls, expected)
    def test_wrong_mode_rejects(self):
        self.rows['clear']['diagnostic']['path']['mode'] = 0
        with self.assertRaises(ValueError): self.run_plan()
        self.assertEqual(self.calls, ['clear'])
    def test_failed_ready_no_live(self):
        self.rows['prepare']['after']['phase'] = 0x80000003
        with self.assertRaises(ValueError): self.run_plan()
        self.assertEqual(self.calls, ['clear', 'prepare'])
    def test_no_frames_not_success(self):
        r = self.rows['live']['register_samples']
        r[-1]['apb']['5c'] = r[0]['apb']['5c']
        with self.assertRaises(ValueError): self.run_plan()
    def test_live_fault_not_success(self):
        self.rows['live']['register_samples'][-1]['apb']['60'] = 1
        with self.assertRaises(ValueError): self.run_plan()
    def test_open_inspection_rejects(self):
        self.rows['prepare']['path']['inspect_enabled'] = True
        with self.assertRaises(ValueError): self.run_plan()
    def test_colorbar_word_rejects(self):
        self.rows['clear']['diagnostic']['word'] = 0x56531001
        with self.assertRaises(ValueError): self.run_plan()

if __name__ == '__main__':
    output = Path(sys.argv[1]); assert not output.exists()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RecoveryTests))
    output.write_text(json.dumps(dict(status='PASS' if result.wasSuccessful() else 'FAIL', tests=result.testsRun,
        helper_sha256=restore.sha(restore.__file__), test_sha256=restore.sha(__file__),
        prepare_sha256=restore.sha(restore.HERE/'video_service_reprepare_preview.py')), indent=2))
    raise SystemExit(not result.wasSuccessful())
