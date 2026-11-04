"""Official independent mapping of thirteen measured control-net replicas."""
from pathlib import Path
from control_command_repair import once
SOURCE=Path(__file__).with_name('cnn_timing_closure_board.py')
code=SOURCE.read_text(encoding='utf8')
code=once(code,'from cnn_run_budget_board import BOARD as PARENT,audit_delta as parent_audit',
               'from cnn_observer_trim_board import BOARD as PARENT,audit_delta as parent_audit')
code=once(code,'from cnn_timing_closure_native import OUT as TRANSFORM',
               'from cnn_sparse_control_native import OUT as TRANSFORM')
code=code.replace('cnn-timing-closure1','cnn-sparse-control1')
code=code.replace('CNN_TIMING_CLOSURE_JOINT_NATIVE_GRAPH_PASS','SPARSE_CONTROL_EXACT_CLONE_GRAPH_PASS')
code=code.replace('CNN_TIMING_CLOSURE_FULL_BOARD_MAP_PASS','CNN_SPARSE_CONTROL_FULL_BOARD_MAP_PASS')
code=code.replace('CNN_TIMING_CLOSURE_FULL_VIDEO_A_MAP_BOUND','CNN_SPARSE_CONTROL_FULL_VIDEO_A_MAP_BOUND')
code=once(code,"r['latency_cycles_added']==0 and r['clones']==307 and r['writer_luts']==156",
               "r['latency_cycles_added']==0 and r['clones']==30 and r['selected_nets']==13 and r['added_kinds']=={'EFX_FF':3,'EFX_LUT4':27}")
code=once(code,"proof['total_added_kinds']","proof['added_kinds']")
code=once(code,"(Path(__file__),TRANSFORM/'result.json'","(Path(__file__),SOURCE,TRANSFORM/'result.json'")
exec(compile(code,str(SOURCE)+':sparse-control','exec'),globals())
