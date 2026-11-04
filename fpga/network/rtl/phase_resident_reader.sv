// Ordered cyclic weights from the existing five verified resident banks.
// No layer-weight copy and no payload RAM here. At most one accepted request
// per bank; local cancellation drains accepted reads without revoking the image.
// Store reset/revocation must use the system's common reset/epoch boundary.
module phase_resident_reader(
 input wire clk,reset,cancel,
 input wire cmd_valid,output wire cmd_ready,input wire [14:0] cmd_weight_offset,
 input wire [7:0] cmd_period,input wire [12:0] cmd_packets,input wire [15:0] cmd_image_tag,
 input wire store_coeff_ready,input wire [15:0] store_image_tag,
 output reg [4:0] fast_read_valid,input wire [4:0] fast_read_ready,
 output reg [44:0] fast_read_address,output wire [79:0] fast_read_tag,
 input wire [4:0] fast_response_valid,output reg [4:0] fast_response_ready,
 input wire [319:0] fast_response_data,input wire [44:0] fast_response_address,
 input wire [79:0] fast_response_tag,input wire [19:0] fast_response_error,
 output wire out_valid,input wire out_ready,output wire [63:0] out_data,output wire [12:0] out_index,
 output wire done_valid,input wire done_ready,output wire busy,output reg error
);
 localparam [1:0] IDLE=0,RUN=1,DRAIN=2,DONE=3;
 reg [1:0] state;reg [4:0] credit;reg request_held;
 reg [2:0] base_bank,issue_bank,expect_bank;reg [8:0] base_address,issue_address,expect_address;
 reg [7:0] period;reg [6:0] issue_phase,expect_phase;
 reg [12:0] packets,issued,received;reg [15:0] image_q;
 wire eligible=store_coeff_ready && store_image_tag==image_q;
 wire available=fast_response_valid[expect_bank] && credit[expect_bank];
 wire response_bad=available && (fast_response_error[expect_bank*4+:4]!=0 ||
  fast_response_tag[expect_bank*16+:16]!=image_q || fast_response_address[expect_bank*9+:9]!=expect_address);
 wire unexpected=|(fast_response_valid & ~credit);
 wire abort_now=state==RUN && (cancel || !eligible || response_bad || unexpected);
 wire active=!reset;
 assign cmd_ready=active && !cancel && state==IDLE;
 assign done_valid=active && state==DONE;assign busy=state!=IDLE;
 assign out_valid=active && state==RUN && !abort_now && available;
 assign out_data=fast_response_data[expect_bank*64+:64];assign out_index=received;
 assign fast_read_tag={5{image_q}};
 wire pop=out_valid && out_ready;
 wire can_issue=issued<packets && (!credit[issue_bank] || (pop && expect_bank==issue_bank));
 wire [4:0] take=fast_read_valid & fast_read_ready,release_word=fast_response_valid & fast_response_ready;
 wire issue_fire=|take;
 always @*begin
  fast_read_valid=0;fast_read_address=0;fast_response_ready=0;
  if(active)begin
   // Once exposed and held, complete that read even when cancellation starts.
   if((state==RUN && !abort_now && can_issue) || ((state==DRAIN || abort_now) && request_held))begin
    fast_read_valid[issue_bank]=1;fast_read_address[issue_bank*9+:9]=issue_address;
   end
   if(state==RUN && !abort_now && available)fast_response_ready[expect_bank]=out_ready;
   if(state==DRAIN || abort_now)fast_response_ready=5'b11111;
  end
 end
 wire [11:0] command_word=cmd_weight_offset[14:3];
 wire good=cmd_weight_offset[2:0]==0 && cmd_weight_offset>=1792 &&
  ({1'b0,cmd_weight_offset}+{5'd0,cmd_period,3'd0})<=12608 &&
  (cmd_period==8 || cmd_period==9 || cmd_period==16 || cmd_period==27 || cmd_period==32 || cmd_period==64 || cmd_period==128) &&
  cmd_packets!=0 && store_coeff_ready && cmd_image_tag==store_image_tag;
 always @(posedge clk)begin
  if(reset)begin
   state<=IDLE;credit<=0;request_held<=0;base_bank<=0;issue_bank<=0;expect_bank<=0;
   base_address<=0;issue_address<=0;expect_address<=0;period<=0;issue_phase<=0;expect_phase<=0;
   packets<=0;issued<=0;received<=0;image_q<=0;error<=0;
  end else begin
   for(integer b=0;b<5;b=b+1)begin
    if(release_word[b])credit[b]<=0;
    if(take[b])credit[b]<=1;
   end
   if(|(fast_read_valid & ~fast_read_ready))request_held<=1;
   if(issue_fire)begin
    request_held<=0;issued<=issued+1'b1;
    if({1'b0,issue_phase}+1'b1==period)begin issue_phase<=0;issue_bank<=base_bank;issue_address<=base_address;end
    else begin
     issue_phase<=issue_phase+1'b1;
     if(issue_bank==4)begin issue_bank<=0;issue_address<=issue_address+1'b1;end else issue_bank<=issue_bank+1'b1;
    end
   end
   if(pop)begin
    received<=received+1'b1;
    if({1'b0,expect_phase}+1'b1==period)begin expect_phase<=0;expect_bank<=base_bank;expect_address<=base_address;end
    else begin
     expect_phase<=expect_phase+1'b1;
     if(expect_bank==4)begin expect_bank<=0;expect_address<=expect_address+1'b1;end else expect_bank<=expect_bank+1'b1;
    end
   end
   if(abort_now)begin state<=DRAIN;error<=1;end
   else case(state)
    IDLE:if(cmd_valid && cmd_ready)begin
     state<=good ? RUN : DONE;error<=!good;period<=cmd_period;packets<=cmd_packets;image_q<=cmd_image_tag;
     issued<=0;received<=0;issue_phase<=0;expect_phase<=0;credit<=0;request_held<=0;
     base_bank<=3'(command_word%5);issue_bank<=3'(command_word%5);expect_bank<=3'(command_word%5);
     base_address<=9'(command_word/5);issue_address<=9'(command_word/5);expect_address<=9'(command_word/5);
    end
    RUN:if(received==packets && credit==0 && !request_held)state<=DONE;
    DRAIN:if(credit==0 && !request_held && !cancel)state<=DONE;
    DONE:if(done_valid && done_ready)state<=IDLE;
    default:begin state<=IDLE;error<=1;end
   endcase
  end
 end
endmodule
