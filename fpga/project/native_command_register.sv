// One registered, non-posted native-port transaction. The DMA transfers
// ownership at capture, then waits for the actual native response. An offered
// request is never withdrawn on timeout/fault; only the common reset clears it.
// This prevents live DMA source/address decode reaching video ready/reset logic
// through the entire native combinational command decoder in one clock.
module native_command_register(
 input wire clk,reset,
 input wire s_valid,output wire s_ready,input wire s_write,
 input wire[31:0]s_address,s_data,
 output wire s_response_valid,output wire s_response_error,output wire[31:0]s_response_data,
 output wire m_valid,output wire m_write,output wire[31:0]m_address,m_data,
 input wire m_ready,m_response_valid,m_response_error,input wire[31:0]m_response_data,
 output reg protocol_fault
);
 reg pending,accepted;
 control_dsp_register #(.WIDTH(65)) payload_storage(.clk(clk),.reset(reset),
  .enable(s_valid && s_ready),.data({s_write,s_address,s_data}),.q({m_write,m_address,m_data}));
 wire take=m_valid && m_ready;
 wire owned_reply=pending && (accepted || take);
 assign s_ready=!reset && !pending && !protocol_fault;
 assign m_valid=!reset && pending && !accepted;
 assign s_response_valid=!reset && m_response_valid && owned_reply;
 assign s_response_error=m_response_error;
 assign s_response_data=m_response_data;
 always @(posedge clk)begin
  if(reset)begin pending<=0;accepted<=0;protocol_fault<=0;end
  else begin
   if(m_response_valid && !owned_reply)protocol_fault<=1;
   if(s_valid && s_ready)begin
    pending<=1;accepted<=0;
   end
   if(take)accepted<=1;
   if(s_response_valid)begin pending<=0;accepted<=0;end
  end
 end
endmodule
