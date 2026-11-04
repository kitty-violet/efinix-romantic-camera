"""Capture the stalled command boundary without changing the frozen DUT."""
import json
import validate_registered_board_cluster as gate

gate.RUN=gate.ROOT/'tmp/frame-execution/split-registered-cluster-diag1'
original_once=gate.once

def with_diagnostic(text,before,after):
    out=original_once(text,before,after)
    if before=='row_pair_arena6 arena(.clk':
        needle='    fail("command watchdog");'
        diagnostic='''    $display("START_DIAG adapter=%0d video=%0d ready=%b parameter=%b localcancel=%b abort=%b load_busy=%b load_done=%b group=%0d",dut.state,dut.engine.state,dut.video_ready,dut.parameter_ready,dut.local_cancel,dut.abort_event,dut.load_busy,dut.load_done,dut.group_q);
    $display("MAC_DIAG state=%0d active=%b srcready=%b weights_ready=%b weight_begin=%b weight_valid=%b source_running=%b source_done=%b source_pending=%b buffer=%0d source_error=%b",dut.engine.mac.state,dut.engine.mac.active,dut.engine.mac.srcready,dut.engine.mac.weights_ready,dut.engine.mac.weight_begin,dut.engine.mac.weight_valid,dut.engine.mac.source.running,dut.engine.mac.source.done,dut.engine.mac.source.pending,dut.engine.mac.source.buffer_count,dut.engine.mac.source.error);
    $display("RQ_DIAG stored=%0d/%0d/%0d selected=%b preparing=%b header=%b/%b/%b",dut.engine.rq.stored[0],dut.engine.rq.stored[1],dut.engine.rq.stored[2],dut.engine.rq.selected,dut.engine.rq.preparing,header_cache.configured,header_cache.have,hur);
'''+needle
        out=original_once(out,needle,diagnostic)
        declaration=' integer fd,ncases,caseid,off,width,height,groups_in,cin_groups,mask_value,ox,origin,srcwidth,span,c,mode,g,elapsed,waits,kind,weight_offset;'
        out=original_once(out,declaration,declaration+'''\n integer start_trace_count=0;
 reg [15:0] start_trace_previous=16'hffff;
 wire [15:0] start_trace_view={dut.state,dut.engine.state,dut.engine.mac.state,dut.video_ready,dut.parameter_ready,dut.engine.mac.srcready,dut.engine.mac.weights_ready,dut.engine.mac.werror,dut.local_cancel,dut.load_busy,dut.load_done};
 always @(negedge clk)if(caseid==12 && start_trace_count<40 && start_trace_view!==start_trace_previous)begin
  $display("START_EDGE t=%0t view=%h src_running=%b done=%b count=%0d cache_loaded=%b loading=%b error=%b term=%0d load_valid=%b good=%b data=%h",$time,start_trace_view,dut.engine.mac.source.running,dut.engine.mac.source.done,dut.engine.mac.source.buffer_count,dut.engine.mac.cache.loaded,dut.engine.mac.cache.loading,dut.engine.mac.cache.error,dut.engine.mac.cache.term,dut.engine.mac.cache.load_valid,dut.engine.mac.cache.good,dut.engine.mac.cache.load_data);
  start_trace_previous=start_trace_view;start_trace_count=start_trace_count+1;
 end''')
    return out

if __name__=='__main__':
    gate.once=with_diagnostic
    manifest=gate.RUN.parent/(gate.RUN.name+'-diagnostic.json')
    assert not manifest.exists()
    manifest.write_text(json.dumps(dict(wrapper_sha256=gate.sha(__file__),
        original_runner_sha256=gate.sha(gate.__file__),board_inputs_sha256=gate.sha(gate.BOARD/'inputs.json'),
        scope='Only three diagnostic displays at the existing watchdog; no DUT or acceptance change.',hardware_access=False),indent=2))
    gate.main()
