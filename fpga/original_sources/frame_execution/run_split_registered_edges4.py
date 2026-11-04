"""Align phase-coverage checks with the frozen production job-selection edge.

Both reference and candidate matched until the older test incorrectly called
the first reservation edge a correction edge. Check reservation, first pass,
final pass and queued results explicitly; preserve all arithmetic/miter checks.
"""
import validate_split_registered_edges3 as gate
gate.RUN=gate.ROOT/'tmp/frame-execution/split-registered-edges-shorts4'
original=gate.rq_test

def rq_test():
 text=original()
 replacements={
  'delay<=8':'delay<=10',
  'if(delay<=3 && !dut.preparing)':'if(delay>=2 && delay<=4 && !dut.preparing)',
  'if(delay>=4 && dut.preparing)':'if(delay>=5 && dut.preparing)',
  'cancel_phases=8':'cancel_phases=10',
  '   if(!busy)$fatal(1,"CANCEL_NOT_ACTIVE");':'''   if(!busy)$fatal(1,"CANCEL_NOT_ACTIVE");
   if(delay==1 && (!dut.selected || dut.preparing))$fatal(1,"RESERVATION_CANCEL_NOT_ACTIVE");
   if(delay>=9 && !out_valid[0])$fatal(1,"QUEUED_CANCEL_NOT_ACTIVE");
   if({dut.selected,dut.preparing,out_valid}!=={reference.selected,reference.preparing,ref_out_valid})
    $fatal(1,"CANCEL_PHASE_REFERENCE_MISMATCH");'''}
 for before,after in replacements.items():
  assert text.count(before)==1,before;text=text.replace(before,after)
 return text

gate.rq_test=rq_test
if __name__=='__main__':
 try:
  gate.main()
  (gate.RUN/'phase_test_generator.json').write_text(gate.json.dumps(dict(
   phase_adapter_sha256=gate.sha(__file__),base_generator_sha256=gate.sha(gate.__file__),
   reason='Existing production RTL has a reservation edge; explicit coverage now includes it and queued output.'),indent=2))
 except Exception as exc:
  if gate.RUN.exists():(gate.RUN/'failure.json').write_text(gate.json.dumps(dict(error=repr(exc)),indent=2))
  raise
