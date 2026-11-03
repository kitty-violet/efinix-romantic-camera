// Only transports individual transactions. All ID probes, register sequence,
// delays, readback policy and configuration publication are owned by RV32 C.
module camera_service_bridge #(
 parameter integer PRESCALE=32,TIMEOUT_CYCLES=20000
)(
 input wire reset,service_clk,camera_clk,camera_power_ready,
 input wire camera_request_valid,output wire camera_request_ready,
 input wire[32:0]camera_request,
 output wire camera_response_valid,input wire camera_response_ready,
 output wire[3:0]camera_response_error,
 output wire[2:0]camera_response_error_byte,
 output wire[7:0]camera_response_data,
 input wire scl_i,sda_i,output wire scl_o,scl_oen,sda_o,sda_oen
);
 wire command_valid,command_ready;wire[32:0]command;
 wire return_ready;reg return_valid,active;reg[14:0]return_value;
 (* async_reg="true" *)reg[1:0]camera_release;
 wire camera_reset=!camera_release[1];
 (* async_reg="true",syn_srlstyle="registers" *)reg[1:0]power_ready_sync;
 wire power_ready=power_ready_sync[1];
 wire transaction_done,transaction_busy;
 wire[3:0]transaction_error;wire[2:0]transaction_error_byte;
 wire[7:0]transaction_data;
 wire transaction_request=command_valid && command_ready;
 assign command_ready=camera_release[1] && power_ready && !active && !return_valid;
 control_atomic_channel #(.WIDTH(33)) requests(
  .reset(reset),.source_clk(service_clk),.destination_clk(camera_clk),
  .s_valid(camera_request_valid),.s_ready(camera_request_ready),.s_data(camera_request),
  .m_valid(command_valid),.m_ready(command_ready),.m_data(command));
 control_atomic_channel #(.WIDTH(15)) responses(
  .reset(reset),.source_clk(camera_clk),.destination_clk(service_clk),
  .s_valid(return_valid),.s_ready(return_ready),.s_data(return_value),
  .m_valid(camera_response_valid),.m_ready(camera_response_ready),
  .m_data({camera_response_error,camera_response_error_byte,camera_response_data}));
 camera_i2c_transaction #(.PRESCALE(PRESCALE),.TIMEOUT_CYCLES(TIMEOUT_CYCLES)) transaction(
  .clk(camera_clk),.reset_n(camera_release[1] && power_ready),.request(transaction_request),.read_reg(command[32]),
  .device_write(command[15:8]),.write_data(command[7:0]),.reg_address(command[31:16]),
  .busy(transaction_busy),.done(transaction_done),.error(transaction_error),
  .error_byte(transaction_error_byte),.read_data(transaction_data),.*);
 always @(posedge camera_clk or posedge reset)begin
  if(reset)camera_release<=0;else camera_release<={camera_release[0],1'b1};
 end
 always @(posedge camera_clk or posedge camera_reset)begin
  if(camera_reset)power_ready_sync<=0;else power_ready_sync<={power_ready_sync[0],camera_power_ready};
 end
 always @(posedge camera_clk or posedge camera_reset)begin
  if(camera_reset)begin active<=0;return_valid<=0;return_value<=0;end
  else begin
   if(transaction_request)active<=1;
   if(transaction_done && active)begin
    active<=0;return_valid<=1;return_value<={transaction_error,transaction_error_byte,transaction_data};
   end
   if(active && !power_ready)begin active<=0;return_valid<=1;return_value<={4'd2,3'd0,8'd0};end
   if(return_valid && return_ready)return_valid<=0;
  end
 end
endmodule
