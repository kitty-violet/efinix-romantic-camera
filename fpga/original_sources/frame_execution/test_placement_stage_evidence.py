"""A successful stage transition is accepted; stale/tampered evidence is not."""
import json,tempfile,unittest
from pathlib import Path
from placement_stage_evidence import placement_result_version,resolve_version
from run_units import ROOT,sha


class Versions(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='stage-evidence-',dir=ROOT/'tmp')
        self.root=Path(self.temp.name);self.physical=self.root/'physical';self.route=self.root/'route'
        (self.physical/'before_route').mkdir(parents=True);(self.physical/'outflow').mkdir();self.route.mkdir()
        self.current=self.physical/'result.json';self.snapshot=self.physical/'before_route/result.json'
        self.old=dict(status='PLACEMENT_ONLY_FINISHED_NOT_RELEASE',stages=[dict(exit_code=0,phase='detailed_place')])
        self.write(self.snapshot,self.old);(self.route/'route.log').write_text('pnr: PASS')
        (self.physical/'outflow/ti60f225_oob.troutingtraces').write_text('test-only-trace')
        self.stages=[dict(status='finished',exit_code=0,log_sha256=sha(self.route/'route.log'))]
        self.write(self.route/'stages.json',self.stages)
        self.new=dict(status='PHYSICAL_CAPACITY_PROBE_FINISHED',stages=self.old['stages']+self.stages,
                      strict_sta_pass=False,board_download_allowed=False,source_unchanged=True)
        self.write(self.current,self.new)
        self.inputs=dict(physical=str(self.physical),required_evidence={str(self.current):sha(self.snapshot)})
        self.write(self.route/'inputs.json',self.inputs)
        self.result=dict(status='VIDEO_SERVICE_LEGAL_ROUTE_PASS_NOT_RELEASE',inputs_sha256=sha(self.route/'inputs.json'),
                         trace_sha256=sha(self.physical/'outflow/ti60f225_oob.troutingtraces'))
        self.write(self.route/'result.json',self.result)
    def tearDown(self):self.temp.cleanup()
    def write(self,p,x):p.write_text(json.dumps(x))
    def test_valid_transition(self):
        v=placement_result_version(self.route)
        self.assertEqual(resolve_version(self.current,sha(self.snapshot),v),self.snapshot)
        self.assertEqual(resolve_version(self.current,sha(self.current),v),self.current)
    def test_unrelated_path_not_redirected(self):
        v=placement_result_version(self.route);p=self.root/'other.json';p.write_bytes(self.snapshot.read_bytes())
        self.assertEqual(resolve_version(p,sha(p),v),p)
    def test_unknown_hash_not_redirected(self):
        v=placement_result_version(self.route)
        self.assertEqual(resolve_version(self.current,'0'*64,v),self.current)
    def test_changed_snapshot_rejected(self):
        self.snapshot.write_text('{}')
        with self.assertRaises(AssertionError):placement_result_version(self.route)
    def test_changed_current_rejected(self):
        self.new['stages']=self.stages;self.write(self.current,self.new)
        with self.assertRaises(AssertionError):placement_result_version(self.route)
    def test_failed_route_rejected(self):
        self.result['status']='FAIL';self.write(self.route/'result.json',self.result)
        with self.assertRaises(AssertionError):placement_result_version(self.route)
    def test_changed_trace_rejected(self):
        (self.physical/'outflow/ti60f225_oob.troutingtraces').write_text('changed')
        with self.assertRaises(AssertionError):placement_result_version(self.route)
    def test_after_resolution_mutation_rejected(self):
        v=placement_result_version(self.route);self.current.write_text('{}')
        with self.assertRaises(AssertionError):resolve_version(self.current,sha(self.snapshot),v)

if __name__=='__main__':unittest.main()
