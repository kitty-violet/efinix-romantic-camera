"""Use the exact paired mapped-cache test for TERM2 only; all other terms kept."""
from pathlib import Path
template=Path(__file__).with_name('validate_video_service_weight_lookahead.py');code=template.read_text()
changes=[
 ("sv=json.loads(survey.read_text());ffs=set(sv['state_registers'])", """sv=json.loads(survey.read_text());sv['rows']=[r for r in sv['rows'] if r['term']==2]
 assert len(sv['rows'])==1
 from continuous_control_native import connections
 drivers,_=connections(old)
 cc={v for row in sv['rows'] for v in row['comb']}
 ffs={drivers[t] for name in cc for pin,ts in old[name][1].items() if pin not in OUT[old[name][0]] for t in ts if t[0]=='net' and drivers.get(t) not in cc}
 assert ffs<=set(sv['state_registers'])"""),
 ('len(storage)==512 and len(ffs)==29 and len(comb)==94','len(storage)==64 and len(ffs)==8 and len(comb)==16'),
 ('[511:0]','[63:0]'),
 ('checks=checks+1024','checks=checks+128'),
 ('storage=512','storage=64'),
 ('checks=8388608','checks=1048576'),
 ('ACTUAL_MAPPED_WEIGHT_LOOKAHEAD_CACHE_PASS','ACTUAL_MAPPED_WEIGHT_TERM2_CACHE_PASS'),
 ('files=[Path(__file__).resolve(),survey,','files=[Path(__file__).resolve(),template.resolve(),survey,'),
]
for old,new in changes:
 assert old in code,old
 code=code.replace(old,new)
exec(compile(code,str(template)+':term2-only','exec'),globals())
