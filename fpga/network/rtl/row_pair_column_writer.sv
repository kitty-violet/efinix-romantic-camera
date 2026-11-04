// Four RQ streams write six row banks. RGB pairs occupy one128-bit transfer;
// feature results use one64-bit half. No duplicate pixel-expansion pipeline.
module row_pair_column_writer(
 input wire clk,reset,cancel,pair_mode,
 input wire [8:0] pixels,input wire [2:0] rows,groups,input wire [7:0] expected_mask,
 input wire [11:0] row_banks,input wire [39:0] row_bases,input wire [3:0] row_phase,
 input wire [3:0] in_valid,output wire [3:0] in_ready,
 input wire [255:0] in_data,input wire [63:0] in_tag,input wire [31:0] in_mask,in_bias_overflow,in_range_error,
 output wire [5:0] request_valid,input wire [5:0] request_ready,output wire [59:0] request_address,
 output wire [11:0] request_half,output wire [767:0] request_data,
 input wire [5:0] response_valid,output wire [5:0] response_ready,input wire [5:0] response_was_write,
 output wire busy,output reg error,output reg [31:0] writes,acks
);
 wire active=!reset && !cancel && !error;
 wire [3:0] bad;wire [2:0] bank[0:3];wire [9:0] address[0:3];wire [1:0] half[0:3];wire [127:0] data[0:3];
 reg [5:0] held,pending,double_ack;reg [1:0] held_source[0:5],next_source[0:5];
 wire [5:0] fire;wire [1:0] grant[0:5];reg [3:0] ready;
 assign in_ready=ready;assign response_ready={6{active}};assign busy=(|held)||(|pending);
 for(genvar s=0;s<4;s=s+1)begin:SOURCE
  wire [15:0] tag=in_tag[s*16+:16];wire [1:0] y=tag[12:11],g=tag[1:0];
  wire [9:0] x=pair_mode ? {tag[10:2],1'b0}:{1'b0,tag[10:2]};
  wire [10:0] local_x={1'b0,x}+{10'd0,row_phase[y]};
  wire [12:0] off=groups==4 ? {1'b0,local_x[10:1],2'd0}:groups==2 ? {2'd0,local_x[10:1],1'b0}:{3'd0,local_x[10:1]};
  wire [13:0] addr={4'd0,row_bases[y*10+:10]}+{1'b0,off}+{12'd0,g};
  assign bank[s]=row_banks[y*3+:3];assign address[s]=addr[9:0];
  assign half[s]=pair_mode ? (x+10'd1<{1'b0,pixels} ? 2'b11:2'b01):(local_x[0] ? 2'b10:2'b01);
  assign data[s]=pair_mode ? {40'd0,in_data[s*64+24+:24],40'd0,in_data[s*64+:24]}:{2{in_data[s*64+:64]}};
  assign bad[s]=in_valid[s] && (tag[15:13]!=0 || {1'b0,y}>=rows || bank[s]>=6 || pixels==0 || pixels>320 || rows==0 || rows>4 ||
   !(groups==1 || groups==2 || groups==4) || {1'b0,g}>=groups || x>={1'b0,pixels} || addr>=1024 ||
   (|in_bias_overflow[s*8+:8]) || (|in_range_error[s*8+:8]) ||
   (pair_mode ? (g!=0 || groups!=1 || row_phase[y] || expected_mask!=7 || in_mask[s*8+:8]!=8'h3f):in_mask[s*8+:8]!=expected_mask));
 end
 for(genvar b=0;b<6;b=b+1)begin:BANK
  reg found;reg [1:0] selected;integer choice;
  always @*begin
   found=0;selected=0;choice=0;
   for(integer n=0;n<4;n=n+1)begin
    choice=(int'(next_source[b])+n)%4;
    if(!found && in_valid[choice] && bank[choice]==3'(b))begin found=1;selected=2'(choice);end
   end
   if(held[b])begin selected=held_source[b];found=in_valid[selected];end
  end
  assign grant[b]=selected;
  assign request_valid[b]=active && !(|bad) && found && (!pending[b] || response_valid[b]);
  assign request_address[b*10+:10]=address[selected];assign request_half[b*2+:2]=half[selected];assign request_data[b*128+:128]=data[selected];
  assign fire[b]=request_valid[b] && request_ready[b];
  always @(posedge clk)begin
   if(reset || cancel)begin held[b]<=0;pending[b]<=0;double_ack[b]<=0;held_source[b]<=0;next_source[b]<=0;end
   else if(active)begin
    if(request_valid[b] && !request_ready[b])begin held[b]<=1;held_source[b]<=selected;end
    if(response_valid[b])pending[b]<=0;
    if(fire[b])begin held[b]<=0;pending[b]<=1;double_ack[b]<=&half[selected];next_source[b]<=selected+2'd1;end
   end
  end
 end
 integer nw,na;
 always @*begin
  ready=0;nw=0;na=0;
  for(integer b=0;b<6;b=b+1)begin
   if(fire[b])begin ready[grant[b]]=1;nw=nw+(( &half[grant[b]]) ? 2:1);end
   if(response_valid[b] && response_ready[b])na=na+(double_ack[b] ? 2:1);
  end
 end
 always @(posedge clk)begin
  if(reset || cancel)begin error<=0;writes<=0;acks<=0;end
  else if(active)begin
   writes<=writes+32'(nw);acks<=acks+32'(na);
   if(|bad)error<=1;
   for(integer b=0;b<6;b=b+1)begin
    if(response_valid[b] && (!pending[b] || !response_was_write[b]))error<=1;
    if(held[b] && !in_valid[held_source[b]])error<=1;
   end
  end
 end
endmodule
