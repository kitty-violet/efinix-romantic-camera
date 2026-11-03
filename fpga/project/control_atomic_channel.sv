// One-entry bundled-data channel. Reset is global to BOTH endpoints, never
// a local CPU or accelerator restart. Local release is synchronized and each
// endpoint waits until its peer is out of reset before offering a transfer.
// data_hold must be constrained as bundled data against the request synchronizer.
module control_atomic_channel #(parameter integer WIDTH=32)(
 input wire reset,source_clk,destination_clk,
 input wire s_valid,output wire s_ready,input wire[WIDTH-1:0]s_data,
 output wire m_valid,input wire m_ready,output wire[WIDTH-1:0]m_data
);
 (* async_reg="true" *) reg[1:0]source_release,destination_release;
 (* async_reg="true" *) reg[1:0]peer_destination,peer_source;
 (* async_reg="true" *) reg[1:0]request_sync,acknowledge_sync;
 reg request_toggle,acknowledge_toggle;
 wire[WIDTH-1:0]data_hold;
 wire source_reset=!source_release[1];
 wire destination_reset=!destination_release[1];
 assign s_ready=!source_reset && peer_destination[1] &&
                 (acknowledge_sync[1]==request_toggle);
 assign m_valid=!destination_reset && peer_source[1] &&
                 (request_sync[1]!=acknowledge_toggle);
 assign m_data=data_hold;
 control_dsp_register #(.WIDTH(WIDTH),.RST_SYNC(0)) payload_storage(
  .clk(source_clk),.reset(source_reset),.enable(s_valid && s_ready),.data(s_data),.q(data_hold));
 always @(posedge source_clk or posedge reset)
  if(reset)source_release<=0;else source_release<={source_release[0],1'b1};
 always @(posedge destination_clk or posedge reset)
  if(reset)destination_release<=0;else destination_release<={destination_release[0],1'b1};
 always @(posedge source_clk or posedge source_reset)begin
  if(source_reset)begin peer_destination<=0;acknowledge_sync<=0;request_toggle<=0;end
  else begin
   peer_destination<={peer_destination[0],destination_release[1]};
   acknowledge_sync<={acknowledge_sync[0],acknowledge_toggle};
   if(s_valid && s_ready)request_toggle<=~request_toggle;
  end
 end
 always @(posedge destination_clk or posedge destination_reset)begin
  if(destination_reset)begin peer_source<=0;request_sync<=0;acknowledge_toggle<=0;end
  else begin
   peer_source<={peer_source[0],source_release[1]};
   request_sync<={request_sync[0],request_toggle};
   if(m_valid && m_ready)acknowledge_toggle<=request_sync[1];
  end
 end
endmodule
