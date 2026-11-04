// Load one OC8 group from the unchanged five-bank coefficient image.
// PW inserts one zero ninth tap per IC8; RGB stem transposes tap*3+IC
// into IC*9+tap. No weight-image rewrite or extra parameter RAM.
module fixed_weight_loader(
 input wire clk,reset,cancel,
 input wire cmd_valid,output wire cmd_ready,input wire cmd_pw,cmd_stem,
 input wire [2:0] cmd_groups,input wire [1:0] cmd_out_group,
 input wire [14:0] cmd_weight_offset,input wire [15:0] cmd_image_tag,
 input wire store_coeff_ready,input wire [15:0] store_image_tag,
 output wire weight_begin,input wire weight_begin_ready,output wire [2:0] weight_parts,
 output wire weight_valid,input wire weight_ready,output wire [5:0] weight_index,
 output wire [63:0] weight_data,output wire weight_last,
 output reg [4:0] fast_read_valid,input wire [4:0] fast_read_ready,
 output reg [44:0] fast_read_address,output wire [79:0] fast_read_tag,
 input wire [4:0] fast_response_valid,output reg [4:0] fast_response_ready,
 input wire [319:0] fast_response_data,input wire [44:0] fast_response_address,
 input wire [79:0] fast_response_tag,input wire [19:0] fast_response_error,
 output wire done_valid,input wire done_ready,output wire busy,output reg error
);
 localparam IDLE=0,BEGIN_LOAD=1,RUN=2,DRAIN=3,DONE=4,PREPARE=5;
 reg [12:0] first_word_q;
 reg [2:0] state;reg pw,stem,held;reg [2:0] parts;reg [15:0] image;
 reg [4:0] credit;reg [2:0] base_bank,issue_bank,expect_bank;
 reg [8:0] base_address,issue_address,expect_address;
 reg [1:0] issue_phase,expect_phase;reg [3:0] issue_tap,expect_tap;
 reg [5:0] issued,received,total;
 wire eligible=store_coeff_ready && store_image_tag==image;
 wire zero_issue=pw && issue_tap==8,zero_return=pw && expect_tap==8;
 wire available=!zero_return && fast_response_valid[expect_bank] && credit[expect_bank];
 wire bad_response=available && (fast_response_error[expect_bank*4+:4]!=0 ||
  fast_response_address[expect_bank*9+:9]!=expect_address || fast_response_tag[expect_bank*16+:16]!=image);
 wire unexpected=|(fast_response_valid & ~credit);
 wire abort_now=(state==RUN || state==BEGIN_LOAD || state==PREPARE) && (cancel || !eligible || bad_response || unexpected);
 assign cmd_ready=!reset && !cancel && state==IDLE;assign busy=state!=IDLE;
 // The last weight has already been accepted before this boundary.
 // Publish its drained completion without an additional empty DONE cycle.
 wire finishing=state==RUN && received==total && credit==0 && !held && !abort_now;
 assign done_valid=!reset && (state==DONE || finishing);
 assign weight_begin=!reset && !abort_now && state==BEGIN_LOAD;
 assign weight_parts=parts;
 assign weight_index={expect_phase,3'd0}+{4'd0,expect_phase}+{2'd0,expect_tap};
 assign weight_valid=!reset && state==RUN && !abort_now && received<total && (zero_return ? received<issued:available);
 assign weight_data=zero_return ? 64'd0:fast_response_data[expect_bank*64+:64];
 assign weight_last=received+6'd1==total;
 assign fast_read_tag={5{image}};
 wire pop=weight_valid && weight_ready;
 wire can_issue=issued<total && (!credit[issue_bank] || (pop && !zero_return && expect_bank==issue_bank));
 wire [4:0] take=fast_read_valid & fast_read_ready,release_word=fast_response_valid & fast_response_ready;
 wire step_issue=(|take) || (!reset && state==RUN && !abort_now && zero_issue && issued<total);
 always @*begin
  fast_read_valid=0;fast_read_address=0;fast_response_ready=0;
  if(!reset)begin
   if((state==RUN && !abort_now && !zero_issue && can_issue) || ((state==DRAIN || abort_now) && held))begin
    fast_read_valid[issue_bank]=1;fast_read_address[issue_bank*9+:9]=issue_address;
   end
   if(state==RUN && !abort_now && available)fast_response_ready[expect_bank]=weight_ready;
   if(state==DRAIN || abort_now)fast_response_ready=31;
  end
 end
 wire [2:0] requested_parts=cmd_stem ? 3'd3:cmd_pw ? cmd_groups:3'd1;
 wire [6:0] selected_offset=cmd_pw ? (cmd_groups==4 ? {cmd_out_group,5'd0}:cmd_groups==2 ? {1'b0,cmd_out_group,4'd0}:{2'd0,cmd_out_group,3'd0}):7'd0;
 wire [12:0] first_word={1'b0,cmd_weight_offset[14:3]}+{6'd0,selected_offset};
 wire [6:0] source_words=cmd_stem ? 7'd27:cmd_pw ? {1'b0,cmd_groups,3'd0}:7'd9;
 wire good=cmd_weight_offset[2:0]==0 && cmd_weight_offset>=1792 && first_word+{6'd0,source_words}<=1576 &&
  (cmd_groups==1 || cmd_groups==2 || cmd_groups==4) && !(cmd_pw && cmd_stem) &&
  (!cmd_stem || (cmd_groups==1 && cmd_out_group==0)) && store_coeff_ready && cmd_image_tag==store_image_tag;
 wire [3:0] next_issue_base={1'b0,base_bank}+{2'd0,issue_phase}+4'd1;
 wire [3:0] next_expect_base={1'b0,base_bank}+{2'd0,expect_phase}+4'd1;
 always @(posedge clk)begin
  if(reset)begin
   state<=IDLE;pw<=0;stem<=0;held<=0;parts<=0;image<=0;credit<=0;base_bank<=0;issue_bank<=0;expect_bank<=0;
   first_word_q<=0;base_address<=0;issue_address<=0;expect_address<=0;issue_phase<=0;expect_phase<=0;issue_tap<=0;expect_tap<=0;issued<=0;received<=0;total<=0;error<=0;
  end else begin
   for(integer b=0;b<5;b=b+1)begin if(release_word[b])credit[b]<=0;if(take[b])credit[b]<=1;end
   if(|(fast_read_valid & ~fast_read_ready))held<=1;
   if(step_issue)begin
    held<=0;issued<=issued+6'd1;
    if(issue_tap==8)begin issue_tap<=0;issue_phase<=issue_phase+2'd1;end else issue_tap<=issue_tap+4'd1;
    if(!zero_issue)begin
     if(stem && issue_tap==8)begin issue_bank<=next_issue_base>=5 ? 3'(next_issue_base-5):next_issue_base[2:0];issue_address<=base_address+9'(next_issue_base>=5);end
     else if(stem)begin issue_bank<=issue_bank>=2 ? issue_bank-3'd2:issue_bank+3'd3;issue_address<=issue_address+9'(issue_bank>=2);end
     else begin issue_bank<=issue_bank==4 ? 3'd0:issue_bank+3'd1;issue_address<=issue_address+9'(issue_bank==4);end
    end
   end
   if(pop)begin
    received<=received+6'd1;
    if(expect_tap==8)begin expect_tap<=0;expect_phase<=expect_phase+2'd1;end else expect_tap<=expect_tap+4'd1;
    if(!zero_return)begin
     if(stem && expect_tap==8)begin expect_bank<=next_expect_base>=5 ? 3'(next_expect_base-5):next_expect_base[2:0];expect_address<=base_address+9'(next_expect_base>=5);end
     else if(stem)begin expect_bank<=expect_bank>=2 ? expect_bank-3'd2:expect_bank+3'd3;expect_address<=expect_address+9'(expect_bank>=2);end
     else begin expect_bank<=expect_bank==4 ? 3'd0:expect_bank+3'd1;expect_address<=expect_address+9'(expect_bank==4);end
    end
   end
   if(abort_now)begin state<=DRAIN;error<=1;end
   else case(state)
    IDLE:if(cmd_valid && cmd_ready)begin
     state<=good ? PREPARE:DONE;first_word_q<=first_word;error<=!good;pw<=cmd_pw;stem<=cmd_stem;parts<=requested_parts;image<=cmd_image_tag;
     total<=({3'd0,requested_parts}<<3)+{3'd0,requested_parts};issued<=0;received<=0;issue_phase<=0;expect_phase<=0;issue_tap<=0;expect_tap<=0;credit<=0;held<=0;

    end
    // Address arithmetic is evaluated from an accepted command register.
    // One prefetch cycle cuts offset+division away from the outer APB cone.
    PREPARE:begin
     base_bank<=3'(first_word_q%5);issue_bank<=3'(first_word_q%5);expect_bank<=3'(first_word_q%5);
     base_address<=9'(first_word_q/5);issue_address<=9'(first_word_q/5);expect_address<=9'(first_word_q/5);
     state<=BEGIN_LOAD;
    end
    BEGIN_LOAD:if(weight_begin && weight_begin_ready)state<=RUN;
    RUN:if(finishing)state<=done_ready ? IDLE:DONE;
    DRAIN:if(credit==0 && !held && !cancel)state<=DONE;
    DONE:if(done_ready)state<=IDLE;
    default:begin state<=IDLE;error<=1;end
   endcase
  end
 end
endmodule
