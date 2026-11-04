"""Allow only the documented Starter size notice, then prove every pin warning."""
import re
from mapped_pin_audit import audit as pin_audit
def audit(body,net):
 notices=re.findall(r'^# \*\* Warning: Design size of (\d+) statements exceeds ModelSim Altera Starter recommended capacity\.$',body,re.M)
 assert len(notices)<=1
 cleaned=re.sub(r'^# \*\* Warning: Design size of \d+ statements exceeds ModelSim Altera Starter recommended capacity\.$','',body,flags=re.M)
 matches=list(re.finditer(r'Errors: 0, Warnings: (\d+)\b',cleaned))
 assert matches and int(matches[-1][1])>=len(notices)
 m=matches[-1];actual=int(m[1])
 cleaned=cleaned[:m.start(1)]+str(actual-len(notices))+cleaned[m.end(1):]
 result=pin_audit(cleaned,net)
 assert result['total']+len(notices)==actual
 result.update(total_in_raw_log=actual,starter_capacity_notices=[int(n) for n in notices])
 return result

