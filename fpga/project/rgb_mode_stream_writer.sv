// Capture-only poison cancellation is recoverable. Global cancel and AXI or
// marker faults remain fatal. No slot is published until final B + exact EOF.
module rgb_mode_stream_writer #(
 parameter integer MAX_CAPTURE_BURST=32
)(
 input wire clk,reset,cancel,frame_abort,native_mode,
 input wire s_valid,output wire s_ready,input wire [23:0] s_rgb,input wire [15:0] s_frame_id,
 input wire s_sof,s_eol,s_eof,
 output wire frame_request,input wire frame_ready,input wire [31:0] frame_address,
 output wire [15:0] requested_frame_id,
 output wire frame_done,frame_failed,output wire busy,output reg error,
 output wire [31:0] awaddr,output wire [7:0] awlen,
 output wire awvalid,input wire awready,
 output wire [127:0] wdata,output wire [15:0] wstrb,output wire wlast,wvalid,input wire wready,
 input wire [1:0] bresp,input wire bvalid,output wire bready,
 output wire graft_active,output wire [1:0] graft_dma_state,
 output wire [10:0] graft_x,output wire [10:0] graft_y,output wire [15:0] graft_id,
 output wire graft_native,graft_dma_done,graft_dma_failed
);
 localparam integer XW=11,YW=11;
 reg native_latched;
 wire [10:0] last_x=native_latched?11'd1919:11'd1279;
 wire [10:0] last_y=native_latched?11'd1079:11'd719;
 reg active,pixels_done,bad;reg [XW-1:0] x;reg [YW-1:0] y;reg [15:0] frame_id;
 wire dma_start_ready,dma_busy,dma_done,dma_failed,dma_config_error,dma_protocol_error,start_fire;
 wire pack_ready,pack_valid,pack_empty;wire [127:0] packed_data;wire dma_data_ready,dma_awvalid;
 wire aborting=cancel || frame_abort || bad || error;
 // Retain the complete next burst before offering AW, so capture never
 // reserves shared DDR while waiting for additional camera pixels.
 wire dma_input_valid,pack_drain_ready;
 wire [127:0] dma_input_data;
 wire buffer_empty,burst_buffered;
 generate if(MAX_CAPTURE_BURST==1)begin:g_single
  assign dma_input_valid=pack_valid;
  assign dma_input_data=packed_data;
  assign pack_drain_ready=dma_data_ready;
  assign buffer_empty=1'b1;
  assign burst_buffered=pack_valid;
 end else begin:g_burst
  localparam integer DEPTH=2*MAX_CAPTURE_BURST;
  localparam integer PW=$clog2(DEPTH);
  reg [127:0] memory[0:DEPTH-1];
  reg [127:0] read_data;
  reg read_collision;
  reg [PW-1:0] wr_ptr,rd_ptr;
  reg [PW:0] count;
  wire push=pack_valid && pack_drain_ready;
  wire pop=dma_input_valid && dma_data_ready;
  wire [PW-1:0] next_rd=rd_ptr+pop;
  // Synchronous RAM read. Suppress a same-address collision for one cycle;
  // complete-burst reservation gives the RAM time to prefetch before W.
  // Memory itself is never reset; invalid entries cannot reach the DMA.
  always @(posedge clk)begin
   if(push)memory[wr_ptr]<=packed_data;
   read_data<=memory[next_rd];
  end
  always @(posedge clk)begin
   if(reset || start_fire)begin wr_ptr<=0;rd_ptr<=0;count<=0;read_collision<=0;end
   else begin
    read_collision<=push && wr_ptr==next_rd;
    if(push)wr_ptr<=wr_ptr+1'b1;
    if(pop)rd_ptr<=rd_ptr+1'b1;
    case({push,pop})
     2'b10:count<=count+1'b1;
     2'b01:count<=count-1'b1;
     default:begin end
    endcase
   end
  end
  assign pack_drain_ready=count<DEPTH && active && !aborting;
  assign dma_input_valid=count!=0 && !read_collision;
  assign dma_input_data=read_data;
  assign buffer_empty=count==0;
  assign burst_buffered={1'b0,count} >= ({1'b0,awlen}+9'd1);
 end endgenerate
 wire address_eligible=burst_buffered || aborting;
 assign awvalid=dma_awvalid && address_eligible;
 assign frame_request=!reset && !cancel && !frame_abort && !active && !error && s_valid && s_sof && dma_start_ready;
 assign start_fire=frame_request && frame_ready;
 assign requested_frame_id=s_frame_id;
 assign s_ready=!reset && !aborting && active && !pixels_done && pack_ready;
 wire input_fire=s_valid && s_ready;
 wire markers_bad=s_frame_id!=frame_id || s_sof!=(x==0 && y==0) || s_eol!=(x==last_x) || s_eof!=(x==last_x && y==last_y);
 assign busy=active || dma_busy;
 assign frame_done=dma_done && !aborting && pixels_done && pack_empty && buffer_empty;
 assign frame_failed=dma_failed || (dma_done && (!pixels_done || !pack_empty || !buffer_empty || aborting));
 rgb24_stream_pack pack(.clk(clk),.reset(reset || start_fire),
  .s_valid(s_valid && !aborting && active && !pixels_done),.s_ready(pack_ready),.s_rgb(s_rgb),
  .m_valid(pack_valid),.m_ready(pack_drain_ready),.m_data(packed_data),.empty(pack_empty));
 axi_frame_write_dma #(.MAX_BURST_BEATS(MAX_CAPTURE_BURST)) dma(.clk(clk),.rst_n(!reset),
  .start_valid(start_fire),.start_ready(dma_start_ready),.start_addr(frame_address),
  .start_beats(native_mode?32'd388800:32'd172800),.start_frame_id(s_frame_id),
  .s_valid(dma_input_valid),.s_ready(dma_data_ready),.s_data(dma_input_data),.abort(aborting),
  .m_axi_awid(),.m_axi_awaddr(awaddr),.m_axi_awlen(awlen),.m_axi_awsize(),.m_axi_awburst(),.m_axi_awvalid(dma_awvalid),.m_axi_awready(awready && address_eligible),
  .m_axi_wdata(wdata),.m_axi_wstrb(wstrb),.m_axi_wlast(wlast),.m_axi_wvalid(wvalid),.m_axi_wready(wready),
  .m_axi_bid(4'd0),.m_axi_bresp(bresp),.m_axi_bvalid(bvalid),.m_axi_bready(bready),
  .busy(dma_busy),.frame_done(dma_done),.frame_failed(dma_failed),.completed_frame_id(),
  .config_error(dma_config_error),.protocol_error(dma_protocol_error),.graft_state(graft_dma_state));
 always @(posedge clk)begin
  if(reset)begin active<=0;pixels_done<=0;bad<=0;error<=0;x<=0;y<=0;frame_id<=0;native_latched<=0;end
  else begin
   if(start_fire)begin native_latched<=native_mode;active<=1;pixels_done<=0;bad<=0;x<=0;y<=0;frame_id<=s_frame_id;end
   if(input_fire)begin
    if(markers_bad)begin bad<=1;error<=1;end
    if(x==last_x)begin x<=0;if(y==last_y)pixels_done<=1;else y<=y+1'b1;end else x<=x+1'b1;
   end
   if(dma_config_error || dma_protocol_error)begin bad<=1;error<=1;end
   if(cancel && active)begin bad<=1;error<=1;end
   if(frame_abort && active)bad<=1;
   if(dma_done || dma_failed)active<=0;
  end
 end
 initial if(MAX_CAPTURE_BURST<1 || MAX_CAPTURE_BURST>128 || (MAX_CAPTURE_BURST & (MAX_CAPTURE_BURST-1))!=0)$fatal(1,"CAPTURE_BURST_POWER_OF_TWO");
 assign graft_native=native_latched;
 assign graft_dma_done=dma_done;assign graft_dma_failed=dma_failed;
 assign graft_active=active;
 assign graft_x=x;assign graft_y=y;assign graft_id=frame_id;
endmodule
