"""Keep the full-board mapper checks while selecting an independently proved graph."""
from pathlib import Path
from control_command_repair import once

def code_for(variant):
    assert variant in ('minimal','data64')
    source=Path(__file__).with_name('cnn_timing_closure_board.py')
    code=source.read_text()
    code=once(code,'from cnn_timing_closure_native import OUT as TRANSFORM',f"TRANSFORM=ROOT/'tmp/frame-execution/cnn-{variant}-native1'")
    code=once(code,"BOARD=ROOT/'tmp/frame-execution/connected-board-cnn-timing-closure1'",f"BOARD=ROOT/'tmp/frame-execution/connected-board-cnn-{variant}1'")
    code=once(code,"r['status']=='CNN_TIMING_CLOSURE_JOINT_NATIVE_GRAPH_PASS'","r['status']=='CNN_TIMING_SMALL_NATIVE_GRAPH_PASS'")
    code=once(code,"r['clones']==307",f"r['clones']=={0 if variant=='minimal' else 64}")
    compile(code,str(source)+':'+variant,'exec')
    return code
