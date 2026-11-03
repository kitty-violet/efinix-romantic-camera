// One real DDR reader for either full style or left-original/right-style.
// Both addresses and frame_id are captured together from the paired owner.
// Fixed RGB24 packing, short bounded bursts, no per-pixel metadata RAM.
module rgb_mode_ddr_reader #(
 parameter integer MAX_READ_BURST=128
)(
 input wire clk,reset,
 input wire start_valid,output wire start_ready,
 input wire [31:0] input_address,output_address,input wire [15:0] frame_id,
 input wire compare_mode,native_mode,input wire[31:0] native_address,
 // Credits cover a complete native burst plus unpack/bypass/header latency.
 input wire native_room_short,native_room_long,
 output wire native_active,
 output wire m_valid,input wire m_ready,output wire [23:0] m_rgb,
 output wire [15:0] m_frame_id,output wire m_sof,m_eol,m_eof,
 output wire [31:0] araddr,output wire [7:0] arlen,output wire arvalid,input wire arready,
 input wire [127:0] rdata,input wire [1:0] rresp,input wire rlast,rvalid,output wire rready,
 output reg frame_done,output reg error,output wire busy,
 output wire[2:0]graft_state,output wire[31:0]graft_line_offset,
 output wire[10:0]graft_x,graft_y,output wire[8:0]graft_remaining,graft_planned
);
 localparam [2:0] IDLE=0,AR=1,R=2,FLUSH=3,RESERVE=4,ROW_WAIT=5;
 localparam [2:0] NEXT_BURST=(MAX_READ_BURST==1) ? RESERVE:AR;
 localparam integer BURST_WIDTH=$clog2(MAX_READ_BURST+1);
 reg [2:0] state;reg [31:0] in_base,out_base,address,line_offset;
 reg [15:0] tag;reg compare,right_half,original;
 assign native_active=original;
 wire[10:0] last_x=original?11'd1919:11'd1279;
 wire[10:0] last_y=original?11'd1079:11'd719;
 wire[31:0] stride=original?32'd5760:32'd3840;
 wire[8:0] line_beats=original?9'd360:(compare?9'd120:9'd240);
 reg [8:0] remaining;reg [BURST_WIDTH-1:0] beat_index,burst_beats;
 reg [10:0] line_y;reg [10:0] pixel_x;reg [10:0] pixel_y;
 reg native_long;
 reg bad_burst;wire unpack_ready,unpack_valid,unpack_empty;
 wire [12:0] to_4k=13'd4096-{1'b0,address[11:0]};
 wire [8:0] available=to_4k[12:4];
 wire [8:0] native_limit=native_long ? 9'((MAX_READ_BURST<128)?MAX_READ_BURST:128) : 9'((MAX_READ_BURST<32)?MAX_READ_BURST:32);
 wire [8:0] limit_beats=original ? native_limit:9'(MAX_READ_BURST);
 wire [8:0] bounded=remaining<limit_beats ? remaining:limit_beats;
 wire [8:0] planned=bounded<available ? bounded:available;
 assign graft_state=state;assign graft_line_offset=line_offset;
 assign graft_x=pixel_x;assign graft_y=pixel_y;assign graft_remaining=remaining;assign graft_planned=planned;
 assign start_ready=state==IDLE && !error && !reset;
 wire start_fire=start_valid && start_ready;
 assign busy=state!=IDLE;
 assign araddr=address;assign arlen=planned[7:0]-1'b1;
 assign arvalid=state==AR && !reset;
 wire expected_last=beat_index==burst_beats-1'b1;
 wire beat_bad=rresp!=0 || rlast!=expected_last;
 assign rready=state==R && (bad_burst || error || unpack_ready);
 wire rfire=rvalid && rready;
 // A successful frame drains the unpacker exactly (720p RGB24 is beat aligned).
 // IDLE with no error therefore already owns an empty unpacker; a new frame
 // does not need a combinational reset from unrelated CNN admission logic.
 rgb24_stream_unpack unpack(.clk(clk),.reset(reset),
  .s_valid(rvalid && state==R && !bad_burst && !error && !beat_bad),.s_ready(unpack_ready),.s_data(rdata),
  .m_valid(unpack_valid),.m_ready(m_ready && !error),.m_rgb(m_rgb),.empty(unpack_empty));
 assign m_valid=unpack_valid && !error;
 assign m_frame_id=tag;assign m_sof=pixel_x==0 && pixel_y==0;
 assign m_eol=pixel_x==last_x;assign m_eof=pixel_x==last_x && pixel_y==last_y;
 always @(posedge clk)begin
  if(reset)begin
   state<=IDLE;in_base<=0;out_base<=0;address<=0;line_offset<=0;
   native_long<=0;tag<=0;compare<=0;right_half<=0;original<=0;remaining<=0;beat_index<=0;burst_beats<=0;
   line_y<=0;pixel_x<=0;pixel_y<=0;bad_burst<=0;frame_done<=0;error<=0;
  end else begin
   frame_done<=0;
   if(start_fire)begin
    if(native_mode ? (native_address!=32'h06c00000 && native_address!=32'h07200000 && native_address!=32'h07800000) :
      ((input_address==32'h01c00000) ? (output_address!=32'h01800000) : (input_address[21:0]!=0 || output_address[21:0]!=0 || input_address>=32'h00c00000 || output_address!=input_address+32'h00c00000)))error<=1;
    else begin
     in_base<=input_address;out_base<=native_mode?native_address:output_address;tag<=frame_id;compare<=compare_mode && !native_mode;original<=native_mode;
     address<=native_mode?native_address:(compare_mode ? input_address:output_address);
     line_offset<=0;right_half<=0;remaining<=native_mode?9'd360:(compare_mode ? 9'd120:9'd240);
     line_y<=0;pixel_x<=0;pixel_y<=0;bad_burst<=0;state<=ROW_WAIT;
    end
   end
   // In single-beat mode reserve the whole response capacity BEFORE AR.
   // No new bytes can arrive until R, so downstream stalls cannot revoke
   // an offered address or hold the shared DDR read owner.
   // The actual upscaler accepts a complete line, then pauses for H/V work.
   // Finish delivering the old line and wait for next-line admission BEFORE
   // claiming DDR. No line data can move while waiting, so the admitted
   // upscaler stays in LOAD through the address handshake.
   if(state==ROW_WAIT && pixel_y==line_y && unpack_empty && m_ready)
    state<=original ? RESERVE:NEXT_BURST;
   // Do not claim the shared read owner until the whole response can drain.
   // Latch the selected length BEFORE ARVALID: room changes never revoke or
   // resize an offered address. 32-beat tails can cross the 3840-pixel HDMI
   // prefill threshold in the existing 4096-token FIFO; whole-row admission
   // would stop below that threshold and deadlock startup.
   if(state==RESERVE)begin
    if(original)begin
     if(m_ready && native_room_short)begin native_long<=native_room_long;state<=AR;end
    end else if(MAX_READ_BURST!=1 || unpack_ready)state<=AR;
   end
   if(arvalid && arready)begin state<=R;burst_beats<=BURST_WIDTH'(planned);beat_index<=0;bad_burst<=0;end
   if(rfire)begin
    if(beat_bad)begin error<=1;bad_burst<=1;end
    if(!bad_burst && !error && !beat_bad)begin address<=address+32'd16;remaining<=remaining-1'b1;end
    if(rlast)begin
     if(bad_burst || error || beat_bad)state<=IDLE;
     else if(remaining!=1)state<=original ? RESERVE:NEXT_BURST;
     else if(compare && !right_half)begin
      right_half<=1;remaining<=120;address<=out_base+line_offset+32'd1920;state<=NEXT_BURST;
     end else if(line_y!=last_y)begin
      line_y<=line_y+1'b1;line_offset<=line_offset+stride;right_half<=0;
      remaining<=line_beats;
      address<=(compare ? in_base:out_base)+line_offset+stride;state<=ROW_WAIT;
     end else state<=FLUSH;
    end else beat_index<=beat_index+1'b1;
   end
   if(m_valid && m_ready)begin
    if(pixel_x==last_x)begin pixel_x<=0;if(pixel_y!=last_y)pixel_y<=pixel_y+1'b1;end
    else pixel_x<=pixel_x+1'b1;
    if(m_eof)begin
     if(state!=FLUSH)error<=1;
     else begin frame_done<=1;state<=IDLE;end
    end
   end
  end
 end
 initial if(MAX_READ_BURST<1 || MAX_READ_BURST>256)$fatal(1,"AXI burst range 1..256 required");
endmodule
