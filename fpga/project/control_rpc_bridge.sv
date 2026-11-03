// APB (CPU clock) to atomic request/response (service clock).
// Global reset reaches both mailbox endpoints. CPU-only reset does not cancel
// a service transaction: its reply is drained and permission is revoked.
module control_rpc_bridge #(parameter integer RPC_TIMEOUT=1048576)(
 input wire reset,cpu_restart,cpu_clk,service_clk,
 input wire psel,penable,pwrite,input wire[15:0]paddr,input wire[31:0]pwdata,
 output wire pready,pslverror,output wire[31:0]prdata,
 output wire request_valid,input wire request_ready,
 output wire request_write,output wire[15:0]request_address,
 output wire[31:0]request_data,
 input wire response_valid,output wire response_ready,
 input wire response_error,input wire[31:0]response_data,
 output wire protocol_error
);
 wire[15:0]request_sequence;
 reg busy,service_busy,sequence_fault,unexpected_response;
 reg abandoned,restart_fault,cpu_seen_active,access_fault;
 localparam integer WAIT_BITS=$clog2(RPC_TIMEOUT+1);
 reg[WAIT_BITS-1:0]wait_count;
 (* async_reg="true" *)reg[1:0]response_fault_sync;
 reg[15:0]service_sequence;
 wire[31:0]timer;
 (* async_reg="true",syn_srlstyle="registers" *)reg[1:0]cpu_release,service_release;
 wire cpu_reset=!cpu_release[1],service_reset=!service_release[1];
 always @(posedge cpu_clk or posedge reset)
  if(reset)cpu_release<=0;else cpu_release<={cpu_release[0],1'b1};
 always @(posedge service_clk or posedge reset)
  if(reset)service_release<=0;else service_release<={service_release[0],1'b1};
 wire access=psel && penable;
 wire local_timer=paddr==16'h0078;
 wire tx_ready,rx_valid,rx_ready,return_ready,command_valid,command_ready;
 wire[64:0]command;
 wire[48:0]response;
 wire issue=access && !local_timer && !busy && !cpu_restart;
 wire timed_out=busy && !abandoned && wait_count==RPC_TIMEOUT;
 wire accepted=request_valid && request_ready;
 wire returned=response_valid && response_ready;
 assign request_valid=command_valid && !service_busy;
 assign command_ready=request_ready && !service_busy;
 assign {request_write,request_address,request_data}=command[48:0];
 assign response_ready=return_ready && (service_busy || accepted);
 wire[15:0]reply_sequence=service_busy?service_sequence:command[64:49];
 control_atomic_channel #(.WIDTH(65))requests(
  .reset(reset),.source_clk(cpu_clk),.s_valid(issue),.s_ready(tx_ready),
  .s_data({request_sequence,pwrite,paddr,pwdata}),
  .destination_clk(service_clk),.m_valid(command_valid),.m_ready(command_ready),
  .m_data(command));
 control_atomic_channel #(.WIDTH(49))responses(
  .reset(reset),.source_clk(service_clk),.s_valid(response_valid && (service_busy || accepted)),
  .s_ready(return_ready),.s_data({reply_sequence,response_error,response_data}),
  .destination_clk(cpu_clk),
  .m_valid(rx_valid),.m_ready(rx_ready),.m_data(response));
 control_dsp_counter #(.WIDTH(32),.RST_SYNC(0)) timer_storage(
  .clk(cpu_clk),.reset(cpu_reset),.enable(1'b1),.q(timer));
 control_dsp_counter #(.WIDTH(16),.RST_SYNC(0)) sequence_storage(
  .clk(cpu_clk),.reset(cpu_reset),.enable(busy && rx_valid && rx_ready),.q(request_sequence));
 wire reply_match=response[48:33]==request_sequence;
 assign pready=access && !cpu_restart && !abandoned &&
  ((local_timer && !busy) || (busy && (rx_valid || timed_out)));
 assign pslverror=access && ((local_timer && pwrite) ||
  timed_out || (busy && rx_valid && (response[32] || !reply_match)));
 assign prdata=local_timer?timer:timed_out?32'd0:response[31:0];
 assign rx_ready=busy && (access || abandoned || cpu_restart);
 assign protocol_error=response_fault_sync[1] || sequence_fault || restart_fault || access_fault;
 always @(posedge cpu_clk or posedge cpu_reset)begin
  if(cpu_reset)begin
   busy<=0;sequence_fault<=0;response_fault_sync<=0;
   abandoned<=0;restart_fault<=0;cpu_seen_active<=0;access_fault<=0;wait_count<=0;
  end
  else begin
   response_fault_sync<={response_fault_sync[0],unexpected_response};

   if(!cpu_restart)cpu_seen_active<=1;
   if(cpu_restart && cpu_seen_active)begin restart_fault<=1;if(busy)abandoned<=1;end
   if(access && local_timer && pwrite)access_fault<=1;
   if(issue && tx_ready)begin busy<=1;abandoned<=0;wait_count<=0;end
   if(busy && !timed_out && !abandoned)wait_count<=wait_count+1'b1;
   if(timed_out)begin abandoned<=1;access_fault<=1;end
   if(busy && rx_valid && rx_ready)begin
    busy<=0;abandoned<=0;
    if(!reply_match)sequence_fault<=1;
    if(response[32])access_fault<=1;
   end
  end
 end
 always @(posedge service_clk or posedge service_reset)begin
  if(service_reset)begin service_busy<=0;service_sequence<=0;unexpected_response<=0;end
  else begin
   if(response_valid && !(service_busy || accepted))unexpected_response<=1;
   if(accepted)begin service_busy<=1;service_sequence<=command[64:49];end
   if(returned)service_busy<=0;
  end
 end
endmodule
