"""Read-only taps on the frozen, real control plane. No command/clock changes."""
import re


def replace_once(text, before, after):
    assert text.count(before) == 1, before[:100]
    return text.replace(before, after, 1)


def instrument_bridge(text):
    text=replace_once(text,'module JtagBridgeNoTap (\n',
        'module JtagBridgeNoTap (\n  input wire [191:0] diagnostic_payload,\n  input wire [7:0] diagnostic_levels,\n')
    needle='  FlowCCUnsafeByToggle flowCCUnsafeByToggle_1 ('
    text=replace_once(text,needle,'''  wire diagnostic_tdo;
  jtag_bringup_snapshot observation(
    .system_clk(io_systemClk),.tck(jtagCtrl_tck),
    .request_capture(jtag_wrapper_sendCapture && jtag_wrapper_headerNext==2),
    .poll_capture(jtag_wrapper_sendCapture && jtag_wrapper_headerNext==3),
    .shift(jtag_wrapper_sendShift && jtag_wrapper_header[1]),
    .probe_payload(diagnostic_payload),.asynchronous_levels(diagnostic_levels),.tdo(diagnostic_tdo));
'''+needle)
    needle='''    if(when_JtagTapInstructions_l209_1) begin
      jtag_wrapper_ctrl_tdo = jtag_readArea_ctrl_tdo;
    end'''
    return replace_once(text,needle,needle+'\n    if(jtag_wrapper_header[1]) jtag_wrapper_ctrl_tdo = diagnostic_tdo;')


TAPS='''
 // Read-only, configuration-initialized evidence survives functional reset.
 // These signals cannot issue APB/AXI requests or change boot permissions.
 reg [31:0] observation_ticks=0;
 reg [7:0] observation_bits=0,observation_last=0,observation_accepts=0,observation_replies=0;
 reg [31:0] observation_address=0,observation_reply=0;
 always @(posedge clk) begin
  observation_ticks<=observation_ticks+1'b1;
  if(!reset && remote_valid) observation_bits<=observation_bits+1'b1;
  if(!reset && remote_valid && remote_last) observation_last<=observation_last+1'b1;
  if(host_valid && host_ready) begin
   observation_accepts<=observation_accepts+1'b1;
   observation_address<=host_address;
  end
  if(host_rsp_valid) begin
   observation_replies<=observation_replies+1'b1;
   observation_reply<=host_rsp_data;
  end
 end
 wire [31:0] observation_flags={6'd0,boot_fault,io_apbSlave_0_PSLVERROR,
  io_apbSlave_0_PREADY,io_apbSlave_0_PENABLE,io_apbSlave_0_PSEL,
  host_owner,fatal_bus_error,boot_terminal,boot_failed,boot_ready,boot_valid,
  host_rsp_valid,host_ready,host_valid,reset,state};
 assign observation_payload={diagnostic_board_status,observation_flags,
  observation_reply,observation_address,
  observation_replies,observation_accepts,observation_last,observation_bits,observation_ticks};
'''


def instrument_soc(text):
    before=text
    text=instrument_bridge(text)
    text=replace_once(text,'module cnn_fixed_control_soc (\n',
        'module cnn_fixed_control_soc (\n  input wire [31:0] diagnostic_board_status,\n  input wire [6:0] diagnostic_levels,\n')
    needle=' JtagBridgeNoTap system_hardJtag_debug_logic_jtagBridge(\n'
    text=replace_once(text,needle,' wire [191:0] observation_payload;\n'+needle+'''  .diagnostic_payload(observation_payload),
  .diagnostic_levels({io_systemReset,diagnostic_levels}),
''')
    needle=' // These original board outputs were unconnected and their peripheral'
    text=replace_once(text,needle,TAPS+'\n'+needle)
    # The existing functional state machines and vendor command CDC stay intact.
    for name in ('cnn_boot_sequencer','SystemDebugger','FlowCCUnsafeByToggle','BufferCC'):
        pattern=r'^module '+name+r'\b.*?^endmodule'
        assert re.findall(pattern,before,re.M|re.S)==re.findall(pattern,text,re.M|re.S)
    return text


def instrument_top(text):
    needle='cnn_fixed_control_soc cpu(.boot_run_allowed(fixed_boot_run_allowed),'
    return replace_once(text,needle,'''cnn_fixed_control_soc cpu(
 .diagnostic_board_status(bringup_ddr_sync),
 .diagnostic_levels({cal_done,process_pll_locked,user_pll_locked,pll_locked,i_pll_locked,i_arstn,w_arstn}),
 .boot_run_allowed(fixed_boot_run_allowed),''')
