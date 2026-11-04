"""Retain complete unchanged CNN data cones; keep 32809 ordinary cells free.

This includes combinational fan-in omitted by name-only membership. Global
clock/reset/enable trees are not recursively frozen. No old route is reused.
"""
from pathlib import Path
from control_command_repair import once
SOURCE=Path(__file__).with_name('cnn_run_budget_engine_locations.py')
code=SOURCE.read_text(encoding='utf8')
code=once(code,'cnn-budget-stable-islands-analysis1.json','cnn-budget-connected-islands-analysis1.json')
code=once(code,"r['scopes']['engine_only']","r['scopes']['cnn_data_cone']")
code=once(code,"s['rejected_components']==0","s['rejected_components']==84")
code=code.replace('9084','27572').replace('51297','32809')
code=code.replace('engine_island','cnn_cone').replace('before_engine_assignments','before_cnn_cone_assignments').replace('engine_generator_sha256','cnn_cone_generator_sha256')
code=once(code,"str(ANALYSIS):sha(ANALYSIS),str(Path(__file__)):sha(__file__)","str(ANALYSIS):sha(ANALYSIS),str(SOURCE):sha(SOURCE),str(Path(__file__)):sha(__file__)")
exec(compile(code,str(SOURCE)+':connected-data-cones','exec'),globals())
