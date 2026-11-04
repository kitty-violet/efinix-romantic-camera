// Dedicated six-row,128-bit pair arena reader. No24-port payload cache.
// This interface is coupled to row_pair_arena6's synchronous read ports:
// a packet fires all needed banks together and retains their BRAM outputs
// until downstream accepts it. Auxiliary users cannot write the A port.
module row_pair_column_reader(
 input wire clk,reset,cancel,
 input wire cmd_valid,output wire cmd_ready,input wire cmd_pair,cmd_rgb,cmd_mode_pw,cmd_nn_view,
 input wire [8:0] cmd_pixels,input wire [2:0] cmd_rows,cmd_groups,cmd_out_groups,input wire [1:0] cmd_group,
 input wire [10:0] cmd_output_x,cmd_source_width,cmd_source_origin,input wire cmd_storage_phase,
 input wire [17:0] cmd_row_banks,input wire [59:0] cmd_row_bases,
 output wire out_valid,input wire out_ready,output wire [767:0] out_data,output wire [12:0] out_index,output wire out_warmup,
 output wire done_valid,input wire done_ready,output wire busy,output reg error,
 output wire [5:0] read_valid,input wire [5:0] read_ready,output wire [59:0] read_address,
 input wire [5:0] read_response,output wire [5:0] read_response_ready,input wire [767:0] read_data,
 output reg [31:0] read_pairs
);
 reg running,done,paired,rgb,is_pw,nn_view,warmup,storage_phase;
 reg [8:0] pixels,batch;reg [2:0] groups;reg [1:0] selected_group;
 reg [10:0] ox,source_width,origin;reg [59:0] bases;reg [17:0] row_map;reg [5:0] needed;
 reg [12:0] next_index,total;reg [6:0] phase,phase_mask;reg [13:0] last_needed;
 reg qvalid,qskip,qhalf_a,qhalf_b;reg [5:0] qmask;reg [12:0] qindex;
 reg [59:0] cb;reg [5:0] cn;reg config_bad;
 wire [3:0] source_rows=cmd_pair ? (cmd_rgb ? {1'b0,cmd_rows}:({1'b0,cmd_rows}<<1)+4'd1):
  {1'b0,cmd_rows}+(cmd_mode_pw ? 4'd0:4'd2);
 always @*begin
  cb=0;cn=0;config_bad=source_rows>6;
  for(integer r=0;r<6;r=r+1)if(4'(r)<source_rows)begin
   if(cmd_row_banks[r*3+:3]>=6)config_bad=1;
   for(integer s=0;s<r;s=s+1)if(cmd_row_banks[r*3+:3]==cmd_row_banks[s*3+:3] &&
    cmd_row_bases[r*10+:10]!=cmd_row_bases[s*10+:10])config_bad=1;
   for(integer bank=0;bank<6;bank=bank+1)if(cmd_row_banks[r*3+:3]==3'(bank))begin
    cn[bank]=1;cb[bank*10+:10]=cmd_row_bases[r*10+:10];
   end
  end
 end
 wire valid_groups=cmd_groups==1 || cmd_groups==2 || cmd_groups==4;
 wire valid_out_groups=cmd_out_groups==1 || cmd_out_groups==2 || cmd_out_groups==4;
 wire [11:0] output_span=cmd_pair && !cmd_rgb ? ({3'd0,cmd_pixels}<<1)-12'd1:{3'd0,cmd_pixels};
 wire good=!config_bad && cmd_pixels>=1 && cmd_pixels<=320 && cmd_rows>=1 && cmd_rows<=4 && valid_groups && valid_out_groups &&
  cmd_source_width>=1 && cmd_source_width<=1280 && cmd_source_origin<(cmd_nn_view ? cmd_source_width>>1:cmd_source_width) &&
  {1'b0,cmd_output_x}+output_span<={1'b0,cmd_source_width} &&
  (cmd_pair ? (!cmd_nn_view && (cmd_rgb ? (cmd_groups==1 && cmd_group==0 && cmd_source_origin<=cmd_output_x):
   (cmd_rows<=2 && cmd_groups<=2 && {1'b0,cmd_group}<cmd_groups && (cmd_output_x==0 ? cmd_source_origin==0:cmd_source_origin<cmd_output_x)))):
   ((!cmd_nn_view || (!cmd_mode_pw && !cmd_source_width[0])) && (cmd_mode_pw ? cmd_source_origin<=cmd_output_x:
    ({1'b0,cmd_group}<cmd_groups && cmd_groups==cmd_out_groups && (cmd_output_x==0 ? cmd_source_origin==0:
     cmd_source_origin<=((cmd_output_x-11'd1)>>(cmd_nn_view ? 1:0)))))));
 wire active=!reset && !cancel && !error;
 assign cmd_ready=active && !running && !done;assign done_valid=done && !reset && !cancel;assign busy=running||done;
 wire [1:0] group_q=(!paired && is_pw) ? phase[4:3] & (groups[1:0]-2'd1):selected_group;
 wire [13:0] normal_raw=is_pw ? {3'd0,ox}+{2'd0,batch,3'd0}+{11'd0,phase[2:0]}:
  warmup ? {3'd0,ox}-14'd1:{3'd0,ox}+{2'd0,batch,3'd0}+{7'd0,(phase==0 ? 7'd0:phase-7'd1)};
 wire signed [14:0] pair_raw=$signed({4'd0,ox})+$signed({1'b0,next_index,1'b0})-(rgb ? 15'sd0:15'sd1);
 wire signed [14:0] raw_a=paired ? pair_raw:(!is_pw && warmup && ox==0) ? 15'sd0:$signed({1'b0,normal_raw});
 wire signed [14:0] raw_b=raw_a+(paired ? 15'sd1:15'sd0);
 wire [10:0] xa=raw_a<0 ? 11'd0:raw_a>=$signed({4'd0,source_width}) ? source_width-11'd1:raw_a[10:0];
 wire [10:0] xb=raw_b<0 ? 11'd0:raw_b>=$signed({4'd0,source_width}) ? source_width-11'd1:raw_b[10:0];
 wire [10:0] pa=nn_view ? xa>>1:xa,pb=nn_view ? xb>>1:xb;
 wire [11:0] la={1'b0,pa}-{1'b0,origin}+{11'd0,storage_phase},lb={1'b0,pb}-{1'b0,origin}+{11'd0,storage_phase};
 wire skip=paired ? (rgb ? next_index>=(({4'd0,pixels}+13'd1)>>1):next_index>{4'd0,pixels}):!warmup && normal_raw>last_needed;
 wire [13:0] offset=groups==4 ? {1'b0,la[11:1],2'd0}:groups==2 ? {2'd0,la[11:1],1'b0}:{3'd0,la[11:1]};
 wire [5:0] bad_addr;
 for(genvar b=0;b<6;b=b+1)begin:ADDRESS
  wire [14:0] addr={5'd0,bases[b*10+:10]}+{1'b0,offset}+{13'd0,group_q};
  assign read_address[b*10+:10]=addr[9:0];
  assign bad_addr[b]=needed[b] && (addr>=1024 || pa<origin || pb<origin || la[11:1]!=lb[11:1]);
 end
 assign out_valid=active && qvalid && ((read_response & qmask)==qmask);
 assign out_index=qindex;assign out_warmup=!paired && !is_pw && qindex==0;
 wire pop=out_valid && out_ready;
 wire can_issue=active && running && next_index<total && (!qvalid || pop);
 wire ready_all=(read_ready & needed)==needed;
 wire issue=can_issue && (skip || (!(|bad_addr) && ready_all));
 assign read_valid=needed & {6{issue && !skip}};
 assign read_response_ready=qmask & {6{pop}};
 for(genvar r=0;r<6;r=r+1)begin:OUTPUT_ROW
  reg [127:0] word_pair;
  always @*begin
   word_pair=0;
   for(integer b=0;b<6;b=b+1)word_pair=word_pair|({128{row_map[r*3+:3]==3'(b)}} & read_data[b*128+:128]);
  end
  assign out_data[r*64+:64]=qskip ? 64'd0:(qhalf_a ? word_pair[127:64]:word_pair[63:0]);
  assign out_data[(r+6)*64+:64]=(r==5 || !paired || qskip) ? 64'd0:(qhalf_b ? word_pair[127:64]:word_pair[63:0]);
 end
 wire [2:0] log_period=(cmd_groups==4 ? 3'd2:cmd_groups==2 ? 3'd1:3'd0)+(cmd_out_groups==4 ? 3'd2:cmd_out_groups==2 ? 3'd1:3'd0);
 wire [12:0] batches=({4'd0,cmd_pixels}+13'd7)>>3;
 wire [12:0] normal_total=cmd_mode_pw ? (batches<<(log_period+3'd3))+{10'd0,(cmd_pixels[2:0]-3'd1)}:
  batches*13'd9+{10'd0,(cmd_pixels[2:0]-3'd1)}+13'd1;
 integer nr;
 always @*begin nr=0;for(integer b=0;b<6;b=b+1)nr=nr+int'(read_valid[b] && read_ready[b]);end
 always @(posedge clk)begin
  if(reset || cancel)begin
   running<=0;done<=0;error<=0;paired<=0;rgb<=0;is_pw<=0;nn_view<=0;warmup<=0;storage_phase<=0;
   pixels<=0;batch<=0;groups<=0;selected_group<=0;ox<=0;source_width<=0;origin<=0;bases<=0;row_map<=0;needed<=0;
   next_index<=0;total<=0;phase<=0;phase_mask<=0;last_needed<=0;qvalid<=0;qskip<=0;qmask<=0;qhalf_a<=0;qhalf_b<=0;qindex<=0;read_pairs<=0;
  end else if(active)begin
   if(done && done_ready)done<=0;
   if(cmd_valid && cmd_ready)begin
    running<=good;done<=!good;error<=!good;paired<=cmd_pair;rgb<=cmd_rgb;is_pw<=cmd_mode_pw;nn_view<=cmd_nn_view;storage_phase<=cmd_storage_phase;
    pixels<=cmd_pixels;groups<=cmd_groups;selected_group<=cmd_group;batch<=0;phase<=0;warmup<=!cmd_mode_pw && !cmd_pair;
    ox<=cmd_output_x;source_width<=cmd_source_width;origin<=cmd_source_origin;bases<=cb;row_map<=cmd_row_banks;needed<=cn;
    next_index<=0;total<=cmd_pair ? (cmd_rgb ? (({4'd0,cmd_pixels}+13'd1)>>1)+13'd7:{4'd0,cmd_pixels}+13'd8):normal_total;
    phase_mask<=cmd_mode_pw ? 7'((8<<log_period)-1):7'd8;
    last_needed<={3'd0,cmd_output_x}+{5'd0,cmd_pixels}-(cmd_mode_pw ? 14'd1:14'd0);
    qvalid<=0;qmask<=0;read_pairs<=0;
   end else read_pairs<=read_pairs+32'(nr);
   if(pop)begin qvalid<=0;qmask<=0;end
   if(can_issue && !skip && |bad_addr)error<=1;
   if(issue)begin
    qvalid<=1;qskip<=skip;qmask<=skip ? 6'd0:needed;qhalf_a<=la[0];qhalf_b<=lb[0];qindex<=next_index;
    next_index<=next_index+13'd1;
    if(warmup)warmup<=0;else if(phase==phase_mask)begin phase<=0;batch<=batch+9'd1;end else phase<=phase+7'd1;
   end
   if(running && next_index==total && !qvalid)begin running<=0;done<=1;end
  end
 end
endmodule
