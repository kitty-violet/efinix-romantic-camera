"""TERM2-only profile; requires its own mapped proofs and routed gates."""
from pathlib import Path
template=Path(__file__).with_name('package_video_service_weight_lookahead.py');code=template.read_text()
changes=[
 ('video-service-native47-weight-lookahead','video-service-native48-weight-term2'),
 ('video-service-mapped-boot26-weight-lookahead','video-service-mapped-boot27-weight-term2'),
 ('video-service-owner-boundary14-weight-lookahead','video-service-owner-boundary15-weight-term2'),
 ('video-service-weight-lookahead-mapped-transport1','video-service-weight-term2-mapped-transport1'),
 ('video-service-weight-lookahead-fixture1','video-service-weight-term2-fixture1'),
 ('video-service-weight-lookahead-data-reset-mapped1','video-service-weight-term2-data-reset-mapped1'),
 ('video-service-weight-lookahead-mapped1','video-service-weight-term2-mapped1'),
 ("ng['weight_lookahead_registers']==24 and ng['weight_lookahead_added']==147 and ng['weight_lookahead_removed']==89", "ng['weight_lookahead_registers']==3 and ng['weight_lookahead_added']==27 and ng['weight_lookahead_removed']==10"),
 ("independent['retained']==54488 and independent['removed']==89 and independent['added']==147", "independent['retained']==54567 and independent['removed']==10 and independent['added']==27"),
 ("independent['changed_srl_addresses']==512 and independent['next_state_boolean_cases']==464", "independent['changed_srl_addresses']==64 and independent['next_state_boolean_cases']==128"),
 ('ACTUAL_MAPPED_WEIGHT_LOOKAHEAD_CACHE_PASS','ACTUAL_MAPPED_WEIGHT_TERM2_CACHE_PASS'),
 ("wt['storage']==512 and wt['checks']==8388608", "wt['storage']==64 and wt['checks']==1048576"),
 ("('validate_video_service_weight_lookahead.py','validate_video_service_weight_data_reset.py'", "('validate_video_service_weight_lookahead.py','validate_video_service_weight_term2.py','package_video_service_weight_lookahead.py','validate_video_service_weight_data_reset.py'"),
 ('==54635','==54594'),
]
for old,new in changes:
 assert old in code,old
 code=code.replace(old,new)
exec(compile(code,str(template)+':term2-proof-profile','exec'),globals())
