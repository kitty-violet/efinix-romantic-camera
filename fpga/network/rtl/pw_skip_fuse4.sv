// Four outstanding row slots, two shared 64-bit RAM return selectors.
// Returns to rows b and b+2 arbitrate fairly; blocked RAM responses retain
// their original port lock until accepted. No read-ahead slot is deleted.
// Use physical port B for skip reads; ordinary PW column streams prefer A.
// Pair actual PW outputs with the retained skip rows. Four one-word read slots
// feed two alignment RQs; two more RQs quantize the signed nine-bit sums.
// All four RQs belong to the SAME mixed pool that produced the PW results.
// MAC results are never written to/read from an intermediate tensor.
module pw_skip_fuse4(
 input wire clk,reset,cancel,enable,main_scaled,
 input wire [8:0] pixels,input wire [2:0] rows,groups,
 input wire [7:0] skip_planes,input wire [39:0] skip_bases,input wire [10:0] skip_delta,
 input wire [3:0] mac_valid,output reg [3:0] mac_ready,input wire [255:0] mac_data,input wire [63:0] mac_tag,
 input wire [31:0] mac_mask,mac_bias_overflow,mac_range_error,
 output wire [3:0] bypass_valid,input wire [3:0] bypass_ready,
 output wire [287:0] bypass_value,output wire [63:0] bypass_tag,output wire [31:0] bypass_mask,
 input wire [1:0] aligned_valid,output wire [1:0] aligned_ready,input wire [127:0] aligned_data,input wire [31:0] aligned_tag,
 input wire [15:0] aligned_mask,aligned_bias_overflow,aligned_range_error,
 input wire [23:0] port_available,output wire [23:0] port_locked,
 output reg [23:0] mem_req_valid,input wire [23:0] mem_req_ready,output reg [239:0] mem_req_address,output reg [383:0] mem_req_tag,
 input wire [23:0] mem_rsp_valid,output wire [23:0] mem_rsp_ready,input wire [1535:0] mem_rsp_data,input wire [383:0] mem_rsp_tag,input wire [23:0] mem_rsp_was_write,
 output wire busy,output reg error,output reg [31:0] paired_words
);
 reg [3:0] waiting,have;reg [63:0] main_q[0:3],skip_q[0:3];reg [15:0] tag_q[0:3];
 reg [11:0] pending,held;reg [1:0] port_owner[0:11];
 reg [1:0] prefer,align_held,align_owner;wire [1:0] chosen;wire [3:0] align_take;
 wire [1:0] qvalid,qempty,qerror,pair_bad;wire active=enable && !reset && !cancel && !error;
 wire [3:0] bad;wire [3:0] target[0:3];wire [9:0] address[0:3];
 // Fused PW bank b always owns logical row b. Retain a tag check so a
 // producer error cannot silently use the specialized direct base connection.
 wire config_bad=groups!=4 || (skip_delta!=0 && skip_delta!=2);
 assign busy=(|waiting)||(|have)||(|pending)||(|held)||!( &qempty);
 assign port_locked={pending[11],1'b0,pending[10],1'b0,pending[9],1'b0,pending[8],1'b0,pending[7],1'b0,pending[6],1'b0,pending[5],1'b0,pending[4],1'b0,pending[3],1'b0,pending[2],1'b0,pending[1],1'b0,pending[0],1'b0} |
  ({held[11],1'b0,held[10],1'b0,held[9],1'b0,held[8],1'b0,held[7],1'b0,held[6],1'b0,held[5],1'b0,held[4],1'b0,held[3],1'b0,held[2],1'b0,held[1],1'b0,held[0],1'b0});
 // Response ready is selected by the shared return arbiters below.
 for(genvar b=0;b<4;b=b+1)begin:TARGET
  wire [15:0] tag=mac_tag[b*16+:16];wire [1:0] y=tag[12:11],g=tag[1:0];wire [8:0] x=tag[10:2];
  wire [9:0] sx={1'b0,x}+{8'd0,skip_delta[1],1'b0};
  wire [10:0] addr={1'b0,skip_bases[b*10+:10]}+{1'b0,sx[9:2],2'd0}+{9'd0,g};
  assign target[b]={skip_planes[b*2+:2],sx[1:0]};assign address[b]=addr[9:0];
  assign bad[b]=mac_valid[b] && (config_bad || tag[15:13]!=0 || y!=2'(b) || {1'b0,y}>=rows || x>=pixels ||
   skip_planes[b*2+:2]>=3 || addr>=1024 || mac_mask[b*8+:8]!=8'hff || (|mac_bias_overflow[b*8+:8]) || (|mac_range_error[b*8+:8]));
 end
 wire [3:0] returned;wire [63:0] return_word[0:3];
 reg [1:0] return_prefer;
 wire [1:0] return_valid;wire [1:0] return_owner[0:1];
 wire [63:0] shared_return_word[0:1];wire [23:0] return_mask[0:1];
 for(genvar s=0;s<2;s=s+1)begin:RETURN_ROUTE
  reg found;reg [3:0] selected_port;reg selected_high;
  always @*begin
   found=0;selected_port=0;selected_high=0;
   for(integer p=0;p<12;p=p+1)begin
    if(active && mem_rsp_valid[p*2+1] && port_locked[p*2+1] && port_owner[p][0]==1'(s) &&
       (!found || (port_owner[p][1]==return_prefer[s] && selected_high!=return_prefer[s])))begin
     found=1;selected_port=4'(p);selected_high=port_owner[p][1];
    end
   end
  end
  wire [23:0] selected_mask;
  reg [63:0] word_q;
  for(genvar p=0;p<12;p=p+1)begin:PORT
   assign selected_mask[p*2+:2]={found && selected_port==4'(p),1'b0};
  end
  always @*begin
   word_q=0;
   for(integer p=0;p<12;p=p+1)word_q=word_q | ({64{selected_mask[p*2+1]}} & mem_rsp_data[p*128+64+:64]);
  end
  assign return_mask[s]=selected_mask;assign return_valid[s]=found;
  assign return_owner[s]={selected_high,1'(s)};assign shared_return_word[s]=word_q;
  always @(posedge clk)begin
   if(reset || cancel)return_prefer[s]<=0;
   else if(found)return_prefer[s]<=!selected_high;
  end
 end
 assign mem_rsp_ready=return_mask[0]|return_mask[1];
 for(genvar b=0;b<4;b=b+1)begin:RETURN_WORD
  assign returned[b]=return_valid[b%2] && return_owner[b%2]==2'(b);
  assign return_word[b]=shared_return_word[b%2];
 end
 reg [1:0] choice[0:11];reg [11:0] request;
 always @*begin
  request=0;mem_req_valid=0;mem_req_address=0;mem_req_tag=0;
  for(integer p=0;p<12;p=p+1)begin
   choice[p]=port_owner[p];
   if(held[p])request[p]=mac_valid[choice[p]];
   else for(integer b=0;b<4;b=b+1)if(!request[p] && mac_valid[b] && target[b]==4'(p) && !waiting[b] && (!have[b] || align_take[b]))begin request[p]=1;choice[p]=2'(b);end
   mem_req_valid[p*2+1]=active && request[p] && !pending[p] && (port_available[p*2+1] || held[p]) && !(|bad);
   mem_req_address[p*20+10+:10]=address[choice[p]];
  end
 end
 always @*begin
  mac_ready=0;
  for(integer p=0;p<12;p=p+1)if(mem_req_valid[p*2+1] && mem_req_ready[p*2+1])mac_ready[choice[p]]=1;
 end
 for(genvar b=0;b<2;b=b+1)begin:ALIGN
  wire select_hi=align_held[b] ? align_owner[b] : (have[b+2] && (!have[b] || prefer[b]));
  wire [1:0] owner={select_hi,1'(b)};wire valid=have[owner];
  assign chosen[b]=select_hi;
  assign bypass_valid[b]=active && valid;
  wire take=bypass_valid[b] && bypass_ready[b];
  assign align_take[b]=take && !select_hi;assign align_take[b+2]=take && select_hi;
  assign bypass_tag[b*16+:16]=tag_q[owner];assign bypass_mask[b*8+:8]=8'hff;
  wire [79:0] qdata;
  wire pop=aligned_valid[b] && aligned_ready[b];
  shallow_result_fifo #(.WIDTH(80)) unscaled(.clk(clk),.reset(reset || cancel),.push(take),
   .in_data({tag_q[owner],(main_scaled ? skip_q[owner]:main_q[owner])}),
   .out_valid(qvalid[b]),.out_ready(pop),.out_data(qdata),.empty(qempty[b]),.error(qerror[b]));
  assign pair_bad[b]=aligned_valid[b] && qvalid[b] && (aligned_tag[b*16+:16]!=qdata[79:64] || aligned_mask[b*8+:8]!=8'hff ||
   (|aligned_bias_overflow[b*8+:8]) || (|aligned_range_error[b*8+:8]));
  assign bypass_valid[b+2]=active && aligned_valid[b] && qvalid[b] && !pair_bad[b];
  assign aligned_ready[b]=active && qvalid[b] && bypass_ready[b+2] && !pair_bad[b];
  assign bypass_tag[(b+2)*16+:16]=qdata[79:64];assign bypass_mask[(b+2)*8+:8]=8'hff;
  for(genvar l=0;l<8;l=l+1)begin:LANE
   wire signed [7:0] v=main_scaled ? main_q[owner][l*8+:8]:skip_q[owner][l*8+:8];
   wire signed [7:0] a=aligned_data[(b*8+l)*8+:8],c=qdata[l*8+:8];
   assign bypass_value[(b*8+l)*9+:9]={v[7],v};
   assign bypass_value[((b+2)*8+l)*9+:9]=$signed({a[7],a})+$signed({c[7],c});
  end
  always @(posedge clk)begin
   if(reset || cancel)begin prefer[b]<=0;align_held[b]<=0;align_owner[b]<=0;end
   else if(active)begin
    if(bypass_valid[b] && !bypass_ready[b])begin align_held[b]<=1;align_owner[b]<=select_hi;end
    if(take)begin align_held[b]<=0;prefer[b]<=!select_hi;end
   end
  end
 end
 always @(posedge clk)begin
  if(reset || cancel)begin
   waiting<=0;have<=0;pending<=0;held<=0;error<=0;paired_words<=0;
   for(integer b=0;b<4;b=b+1)begin main_q[b]<=0;skip_q[b]<=0;tag_q[b]<=0;end
   for(integer p=0;p<12;p=p+1)begin port_owner[p]<=0;end
  end else if(active)begin
   if((|bad)||(|pair_bad)||(|qerror))error<=1;
   paired_words<=paired_words+32'(bypass_valid[2] && bypass_ready[2])+32'(bypass_valid[3] && bypass_ready[3]);
   for(integer b=0;b<4;b=b+1)begin
    if(align_take[b])have[b]<=0;
    if(returned[b])begin waiting[b]<=0;have[b]<=1;skip_q[b]<=return_word[b];end
    if(mac_valid[b] && mac_ready[b])begin waiting[b]<=1;main_q[b]<=mac_data[b*64+:64];tag_q[b]<=mac_tag[b*16+:16];end
   end
   for(integer p=0;p<12;p=p+1)begin
    if(mem_rsp_valid[p*2+1] && mem_rsp_ready[p*2+1])begin
     pending[p]<=0;
     if(!pending[p] || mem_rsp_was_write[p*2+1])error<=1;
    end
    if(mem_req_valid[p*2+1] && !mem_req_ready[p*2+1])begin held[p]<=1;port_owner[p]<=choice[p];end
    if(mem_req_valid[p*2+1] && mem_req_ready[p*2+1])begin
     held[p]<=0;pending[p]<=1;port_owner[p]<=choice[p];
    end
   end
  end
 end
endmodule
