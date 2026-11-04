"""Current schema6 phase/word/epoch next-state exhaustive mapped check."""
from pathlib import Path
source=Path(__file__).with_name('validate_video_service_narrow_counter.py');code=source.read_text()
old="key,g=pickle.loads((board/'service-validation-graph.pkl').read_bytes());assert key[0]==sha(src)"
assert code.count(old)==1
code=code.replace(old,"key,sg,g,sp,ps=pickle.loads((board/'retained-audit-parse.pkl').read_bytes());assert key[1]==sha(src)")
code=code.replace('ps=params(src.read_text());d,_=connections(g)','d,_=connections(g)')
for x,y in [('range(6)','range(7)'),('(phase+1)%6','(phase+1)%7'),('if phase==5:','if phase==6:'),('accepted_requests_per_epoch=96','accepted_requests_per_epoch=112')]:
 assert x in code;code=code.replace(x,y)
exec(compile(code,str(source)+':schema6-current-map','exec'),globals())
