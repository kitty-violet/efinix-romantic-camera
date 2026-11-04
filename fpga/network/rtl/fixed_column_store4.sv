// One four-row result vector, distinct destination banks, fixed C8 groups.
// Keep payload in the upstream queue until every enabled row is accepted.
// Independent bank acceptance/ACK is tracked without another payload copy.
module fixed_column_store4 #(parameter integer REGISTER_ADDRESSES=0)(
 input wire clk,reset,cancel,prepare,
 input wire [8:0] pixels,input wire [2:0] rows,groups,input wire [1:0] group_id,
 input wire [11:0] row_banks,input wire [39:0] row_bases,input wire [3:0] row_phase,
 input wire in_valid,output wire in_ready,input wire [255:0] in_data,input wire [8:0] in_x,input wire in_bad,
 output wire [5:0] request_valid,input wire [5:0] request_ready,output wire [59:0] request_address,
 output wire [11:0] request_half,output wire [767:0] request_data,
 input wire [5:0] response_valid,output wire [5:0] response_ready,input wire [5:0] response_was_write,
 output wire busy,output reg error,output wire [12:0] writes,acks
);
 reg [5:0] accepted,pending;reg [8:0] next_x;
 reg config_bad;reg [5:0] needed;wire [3:0] address_bad;
 wire active=!reset && !cancel && !error;
 always @*begin
  config_bad=pixels==0 || pixels>320 || rows==0 || rows>4 || !(groups==1 || groups==2 || groups==4) || {1'b0,group_id}>=groups;
  needed=0;
  for(integer r=0;r<4;r=r+1)if(3'(r)<rows)begin
   if(row_banks[r*3+:3]>=6)config_bad=1;
   for(integer s=0;s<r;s=s+1)if(row_banks[r*3+:3]==row_banks[s*3+:3])config_bad=1;
   for(integer b=0;b<6;b=b+1)if(row_banks[r*3+:3]==3'(b))needed[b]=1;
  end
 end
 wire [9:0] address[0:3];wire [1:0] half[0:3];
 // Production descriptors remain owned until DONE. Prepare the row address
 // once, before any result, then advance only when the entire vector is taken.
 // The extra bit retains the original out-of-range rejection before any write.
 generate if(REGISTER_ADDRESSES)begin:CURSOR
  for(genvar r=0;r<4;r=r+1)begin:ROW
   reg [10:0] addr;reg high_half;
   always @(posedge clk)begin
    if(prepare)begin addr<={1'b0,row_bases[r*10+:10]}+{9'd0,group_id};high_half<=row_phase[r];end
    else if(reset || cancel)begin addr<=0;high_half<=0;end
    else if(in_valid && in_ready)begin
     high_half<=!high_half;
     if(high_half)addr<=addr+{8'd0,groups};
    end
   end
   assign address[r]=addr[9:0];assign half[r]=high_half ? 2'b10:2'b01;
   assign address_bad[r]=3'(r)<rows && addr[10];
  end
 end else begin:COMBINATIONAL
 for(genvar r=0;r<4;r=r+1)begin:ROW
  // Accepted vectors must match the local sequence counter (checked below).
  wire [9:0] local_x={1'b0,next_x}+{9'd0,row_phase[r]};
  wire [12:0] off=groups==4 ? {2'd0,local_x[9:1],2'd0}:groups==2 ? {3'd0,local_x[9:1],1'b0}:{4'd0,local_x[9:1]};
  wire [13:0] addr={4'd0,row_bases[r*10+:10]}+{1'b0,off}+{12'd0,group_id};
  assign address[r]=addr[9:0];assign half[r]=local_x[0] ? 2'b10:2'b01;
  assign address_bad[r]=3'(r)<rows && addr>=1024;
 end
 end endgenerate
 wire bad=in_bad || config_bad || (|address_bad) || in_x!=next_x || in_x>=pixels;
 wire [5:0] fire=request_valid & request_ready;
 for(genvar b=0;b<6;b=b+1)begin:BANK
  reg [9:0] addr;reg [1:0] halves;
  wire [3:0] row_selected;
  for(genvar r=0;r<4;r=r+1)begin:SELECT
   assign row_selected[r]=3'(r)<rows && row_banks[r*3+:3]==3'(b);
  end
  // Validated distinct banks make this selection one-hot. Remove the serial
  // priority muxes; an unused bank has no enabled write half.
  // Only valid selected banks have enabled write halves. For all other
  // banks the payload is irrelevant; avoid zero-gating 64 bits per bank.
  (* syn_keep = "true" *) wire [63:0] data=row_selected[0] ? in_data[0+:64]:
   row_selected[1] ? in_data[64+:64]:row_selected[2] ? in_data[128+:64]:in_data[192+:64];
  always @*begin
   addr=0;halves=0;
   for(integer r=0;r<4;r=r+1)if(row_selected[r])begin addr=address[r];halves=half[r];end
  end
  assign request_valid[b]=active && in_valid && !bad && needed[b] && !accepted[b] && (!pending[b] || response_valid[b]);
  assign request_address[b*10+:10]=addr;assign request_half[b*2+:2]=halves;assign request_data[b*128+:128]={2{data}};
 end
 assign in_ready=active && !bad && ((accepted | fire) & needed)==needed;
 assign response_ready={6{active}};assign busy=(|accepted)||(|pending);
 // Register the small event counts before wide accumulation. The public
 // view includes the pending events, so requests, counts and DONE keep
 // their original cycles while the ready/cancel cone ends at a 3-bit D.
 integer nw,na;
 reg [12:0] writes_base,acks_base;
 reg [2:0] write_events,ack_events;
 assign writes=writes_base+{10'd0,write_events};
 assign acks=acks_base+{10'd0,ack_events};
 always @(posedge clk)begin
  if(reset || cancel)begin
   writes_base<=0;acks_base<=0;write_events<=0;ack_events<=0;
  end else begin
   writes_base<=writes_base+{10'd0,write_events};
   acks_base<=acks_base+{10'd0,ack_events};
   write_events<=active ? 3'(nw):3'd0;
   ack_events<=active ? 3'(na):3'd0;
  end
 end
 always @*begin nw=0;na=0;for(integer b=0;b<6;b=b+1)begin nw=nw+int'(fire[b]);na=na+int'(response_valid[b] && response_ready[b]);end end
 always @(posedge clk)begin
  if(reset || cancel)begin accepted<=0;pending<=0;next_x<=0;error<=0;end
  else if(active)begin
   pending<=(pending & ~response_valid)|fire;accepted<=accepted|fire;
   if(in_valid && bad)error<=1;
   if(in_valid && in_ready)begin accepted<=0;next_x<=next_x+9'd1;end
   if((|(response_valid & ~pending)) || (|(response_valid & ~response_was_write)) || ((|accepted) && !in_valid))error<=1;
  end
 end
endmodule
