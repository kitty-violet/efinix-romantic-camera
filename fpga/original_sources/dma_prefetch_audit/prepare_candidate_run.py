from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent
tb=(R/'tb_s4r1_frame.sv').read_text('utf8')
assert tb.count('if(production_cycles!=4357918)')==1
tb=tb.replace('  if(production_cycles!=4357918)$fatal(1,"AUDIT_CHANGED_TRACE cycles=%0d",production_cycles);','  // Candidate duration is measured, not asserted against its parent.')
tb=tb.replace('1:category=arready ? 1:2;','1:category=dut.core.dma.arready ? 1:2;')
tb=tb.replace('2:category=!rvalid ? 3:!rready ? 4:5;','2:category=!dut.core.dma.rvalid ? 3:!dut.core.dma.rready ? 4:5;')
(R/'tb_candidate.sv').write_text(tb,'utf8')
runner=(R/'run_trace.py').read_text('utf8')
runner=runner.replace("d=R/('trace'+sys.argv[1])","d=R/('candidate-trace'+sys.argv[1])")
runner=runner.replace("[R/'tb_s4r1_frame.sv']","[R/'tb_candidate.sv']")
runner=runner.replace("(R/'vectors').as_posix()","(R/'candidate-vectors').as_posix()")
(R/'run_candidate.py').write_text(runner,'utf8')
print('CANDIDATE_HARNESS_READY')
