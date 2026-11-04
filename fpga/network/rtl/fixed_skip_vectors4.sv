// Residual pairing owns each branch at its original producer until consumed.
// Scale skip ahead of MAC when main_scaled=0; otherwise scale MAC and pair
// its output with the single skip capture buffer. No duplicate branch FIFO.
module fixed_skip_vectors4 #(parameter integer BASE_WIDTH=10,OFFSET_PREAPPLIED=0,REGISTERED_FEEDBACK_SELECT=0)(
 input wire clk,reset,cancel,enable,main_scaled,
 input wire [8:0] pixels,input wire [2:0] rows,input wire [1:0] group_id,
 input wire [11:0] skip_banks,input wire [4*BASE_WIDTH-1:0] skip_bases,input wire offset_two,
 input wire mac_valid,output wire mac_ready,input wire [255:0] mac_data,input wire [8:0] mac_x,input wire mac_bad,
 output wire align_valid,input wire align_ready,output wire [255:0] align_value,output wire [8:0] align_x,
 input wire aligned_valid,output wire aligned_ready,input wire [255:0] aligned_data,input wire [8:0] aligned_x,input wire aligned_bad,
 output wire post_valid,input wire post_ready,output wire [287:0] post_value,output wire [8:0] post_x,
 output wire [5:0] request_valid,input wire [5:0] request_ready,output wire [59:0] request_address,
 input wire [5:0] response_valid,output wire [5:0] response_ready,input wire [767:0] response_data,input wire [5:0] response_was_write,
 input wire feedback_post_selected,output wire [287:0] feedback_value,
 output wire busy,output reg error
);
 reg skip_held;reg [255:0] skip_data;reg [8:0] read_x,skip_x,issued,completed;reg [5:0] sent;
 reg [3:0] stored;wire active=enable && !reset && !cancel && !error;
 reg config_bad;reg [5:0] needed;
 always @*begin
  needed=0;config_bad=rows==0 || rows>4 || pixels==0 || pixels>320;
  for(integer r=0;r<4;r=r+1)if(3'(r)<rows)begin
   if(skip_banks[r*3+:3]>=6)config_bad=1;
   for(integer s=0;s<r;s=s+1)if(skip_banks[r*3+:3]==skip_banks[s*3+:3])config_bad=1;
   for(integer b=0;b<6;b=b+1)if(skip_banks[r*3+:3]==3'(b))needed[b]=1;
  end
 end
 wire [9:0] sx={1'b0,read_x}+(OFFSET_PREAPPLIED ? 10'd0:{8'd0,offset_two,1'b0});
 wire [3:0] address_bad;wire [9:0] addr[0:3];wire [255:0] skip;
 wire [383:0] halves;
 for(genvar b=0;b<6;b=b+1)begin:HALF
  assign halves[b*64+:64]=sx[0] ? response_data[b*128+64+:64]:response_data[b*128+:64];
 end
 for(genvar r=0;r<4;r=r+1)begin:ROW
  wire [12:0] address={{(13-BASE_WIDTH){1'b0}},skip_bases[r*BASE_WIDTH+:BASE_WIDTH]}+{2'd0,sx[9:1],2'd0}+{11'd0,group_id};
  assign addr[r]=address[9:0];assign address_bad[r]=3'(r)<rows && address>=1024;
  // A six-way binary selector avoids per-bit equality/zero qualification.
  // Inactive logical rows still return zero; invalid banks remain rejected.
  wire [63:0] data=3'(r)<rows && skip_banks[r*3+:3]<6 ? halves[skip_banks[r*3+:3]*64+:64]:64'd0;
  assign skip[r*64+:64]=data;
 end
 wire bad=config_bad || (read_x<pixels && (|address_bad));
 wire input_bad=main_scaled ? (mac_valid && (mac_bad || mac_x!=issued || mac_x>=pixels)):
   (skip_held && (skip_x!=issued || skip_x>=pixels));
 wire pair_available=main_scaled ? skip_held:mac_valid;
 wire [8:0] pair_x=main_scaled ? skip_x:mac_x;
 wire [255:0] pair_data=main_scaled ? skip_data:mac_data;
 wire pair_bad=aligned_valid && (aligned_bad || aligned_x!=completed || aligned_x>=pixels ||
   (pair_available && (aligned_x!=pair_x || (!main_scaled && mac_bad))));
 assign align_valid=active && !bad && !input_bad && issued<pixels && stored<8 && (main_scaled ? mac_valid:skip_held);
 assign align_value=main_scaled ? mac_data:skip_data;assign align_x=issued;
 wire push=align_valid && align_ready;
 assign post_valid=active && aligned_valid && pair_available && stored!=0 && !pair_bad;
 assign aligned_ready=post_valid && post_ready;
 assign post_x=completed;
 wire pop=post_valid && post_ready;
 assign mac_ready=main_scaled ? push:pop;
 wire release_skip=main_scaled ? pop:push;
 wire [5:0] fire=request_valid & request_ready;
 for(genvar b=0;b<6;b=b+1)begin:BANK
  reg [9:0] address;
  always @*begin address=0;for(integer r=0;r<4;r=r+1)address=address | ({10{skip_banks[r*3+:3]==3'(b) && 3'(r)<rows}} & addr[r]);end
  assign request_valid[b]=active && read_x<pixels && !skip_held && !bad && needed[b] && !sent[b];
  assign request_address[b*10+:10]=address;
 end
 // Fused descriptors require four input groups, hence at least four MAC
 // input-vector cycles per output. Start the next skip read from registered
 // empty credit; do not combinationally borrow a same-cycle RQ release.
 // Requests and response acceptance are independent of align/post READY.
 // This preserves one stored vector and adds no RAM or payload register.
 // Capture first, releasing every RAM response before waiting for RQ credit.
 // A blocked output writer therefore cannot form a feedback/BRAM wait cycle.
 wire capture_skip=active && read_x<pixels && !skip_held && !bad && (sent & needed)==needed && (response_valid & needed)==needed;
 assign response_ready=needed & {6{capture_skip}};
 // Align and post never issue together. Share the MAC/skip selection
 // before the adder. This additional bus is qualified by the RQ acceptance,
 // not an independently held ready/valid interface. The public post_value
 // remains unchanged for observers and non-specialized users.
 // Shared production RQ reserves the operation one cycle before issue.
 // Use that reservation for data only. All public ready/valid,
 // cancel, error and credit guards above remain unchanged.
 wire feedback_is_post=REGISTERED_FEEDBACK_SELECT!=0 ? feedback_post_selected:post_ready;
 wire feedback_select;
 EFX_LUT4 #(.LUTMASK(16'h6666)) feedback_selector(
  .I0(main_scaled),.I1(feedback_is_post),.I2(1'b0),.I3(1'b0),.O(feedback_select));
 (* syn_keep = "true" *) wire [255:0] feedback_rhs=feedback_select ? mac_data:skip_data;
 for(genvar l=0;l<32;l=l+1)begin:FEEDBACK
  wire signed [7:0] a=feedback_is_post ? aligned_data[l*8+:8]:8'sd0;
  wire signed [7:0] b=feedback_rhs[l*8+:8];
  assign feedback_value[l*9+:9]=$signed({a[7],a})+$signed({b[7],b});
 end
 for(genvar l=0;l<32;l=l+1)begin:SUM
  wire signed [7:0] a=aligned_data[l*8+:8],b=pair_data[l*8+:8];
  assign post_value[l*9+:9]=$signed({a[7],a})+$signed({b[7],b});
 end
 assign busy=skip_held || (|sent) || stored!=0 || (active && read_x<pixels);
 always @(posedge clk)begin
  if(reset || cancel)begin skip_held<=0;skip_data<=0;read_x<=0;skip_x<=0;sent<=0;stored<=0;issued<=0;completed<=0;error<=0;end
  else if(active)begin
   sent<=sent|fire;
   if(release_skip)skip_held<=0;
   if(capture_skip)begin skip_held<=1;skip_data<=skip;skip_x<=read_x;read_x<=read_x+9'd1;sent<=0;end
   if(push)issued<=issued+9'd1;if(pop)completed<=completed+9'd1;
   case({push,pop})2'b10:stored<=stored+4'd1;2'b01:stored<=stored-4'd1;default:stored<=stored;endcase
   if(bad || input_bad || pair_bad || stored>8 || completed>issued || (|(response_valid & ~sent)) || (|(response_valid & response_was_write)))error<=1;
  end
 end
endmodule
