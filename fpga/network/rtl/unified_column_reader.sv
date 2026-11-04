// Repeat packets retain ownership of the already captured physical port words.
// They share the previous owner's cache, issue no RAM request, and prevent
// younger reservations overwriting their payload. No second 384-bit copy.
// Requests may fire on reservation; output waits for registered response data.
// This moves the bypass across the RAM latency instead of bypassing 24x64 data.
// Independently reserved read packets for normal DW/PW/NN and pair stride/RGB.
// Four metadata slots own disjoint subsets of the same 24 physical payloads.
// Per-port address stays fixed until acceptance; pending identifies its response.
// Four metadata slots share exactly 24 port payload registers. Column ownership
// prevents overwrite; held requests and responses retain owner until completion.
module unified_column_reader(
 input wire clk,reset,cancel,
 input wire cmd_valid,output wire cmd_ready,input wire cmd_pair,cmd_rgb,cmd_mode_pw,cmd_nn_view,
 input wire [8:0] cmd_pixels,input wire [2:0] cmd_rows,cmd_groups,cmd_out_groups,input wire [1:0] cmd_group,
 input wire [10:0] cmd_output_x,cmd_source_width,cmd_source_origin,
 input wire [11:0] cmd_row_planes,input wire [59:0] cmd_row_bases,
 output wire out_valid,input wire out_ready,output wire [767:0] out_data,output wire [12:0] out_index,output wire out_warmup,
 output wire done_valid,input wire done_ready,output wire busy,output reg error,
 input wire [23:0] port_available,output wire [23:0] port_locked,
 output wire [23:0] mem_req_valid,input wire [23:0] mem_req_ready,
 output wire [239:0] mem_req_address,output wire [383:0] mem_req_tag,
 input wire [23:0] mem_rsp_valid,output wire [23:0] mem_rsp_ready,
 input wire [1535:0] mem_rsp_data,input wire [383:0] mem_rsp_tag,input wire [23:0] mem_rsp_was_write,
 output reg [31:0] read_words
);
 reg running,done,paired,rgb,is_pw,nn_view,warmup;
 reg [8:0] pixels,batch;reg [2:0] groups;reg [1:0] selected_group;
 reg [10:0] ox,source_width,origin;reg [59:0] bases;reg [17:0] row_map;reg [5:0] needed;
 reg [12:0] next_index,total;reg [6:0] phase,phase_mask;reg [13:0] last_needed;
 reg [1:0] head,tail;reg [3:0] live,q_skip,q_reuse;
 reg [23:0] q_ports[0:3];reg [1:0] q_a[0:3],q_b[0:3];reg [12:0] q_index[0:3];
 reg [23:0] todo,held,pending,have;
 reg [9:0] request_address[0:23];
 wire [239:0] prepared_address;wire [23:0] sent=24'd0;reg [63:0] payload[0:23];
 reg previous_valid;reg [10:0] previous_x;reg [1:0] previous_group;
 reg [59:0] cb;reg [17:0] cm;reg [5:0] cn;reg config_bad;
 // Fixed B1 allocator invariant: first three logical rows use the first
 // row slot of a plane; sixth row never selects plane0/2 first slots.
 // Reject unsupported descriptors before issuing any physical request.
 wire [3:0] source_rows=cmd_pair ? (cmd_rgb ? {1'b0,cmd_rows}:({1'b0,cmd_rows}<<1)+4'd1) :
  {1'b0,cmd_rows}+(cmd_mode_pw ? 4'd0:4'd2);
 wire layout_bad=cm[0] || (source_rows>1 && cm[3]) || (source_rows>2 && cm[6]) ||
  (source_rows>5 && (cm[15+:3]==0 || cm[15+:3]==4));
 // B1 rows either repeat the immediately preceding clamped/NN row, or
 // advance to the next of three planes. Count physical rows once; this
 // replaces the sequential search/allocate of arbitrary plane/base pairs.
 wire [2:0] ordinal[0:5];
 assign ordinal[0]=0;
 for(genvar r=1;r<6;r=r+1)begin:ROW_ORDINAL
  assign ordinal[r]=ordinal[r-1]+3'(cmd_row_planes[r*2+:2]!=cmd_row_planes[(r-1)*2+:2]);
 end
 always @*begin
  cb=0;cm=0;cn=0;config_bad=source_rows>6;
  for(integer r=0;r<6;r=r+1)if(4'(r)<source_rows)begin
   if(cmd_row_planes[r*2+:2]>=3)config_bad=1;
   if(r>0)begin
    if(cmd_row_planes[r*2+:2]==cmd_row_planes[(r-1)*2+:2])begin
     if(cmd_row_bases[r*10+:10]!=cmd_row_bases[(r-1)*10+:10])config_bad=1;
    end else if(cmd_row_planes[r*2+:2]!=(cmd_row_planes[(r-1)*2+:2]==2 ? 2'd0:cmd_row_planes[(r-1)*2+:2]+2'd1))config_bad=1;
   end
   cm[r*3+:3]={cmd_row_planes[r*2+:2],(ordinal[r]>=3)};
   for(integer slot=0;slot<6;slot=slot+1)if(cm[r*3+:3]==3'(slot))begin
    cn[slot]=1;cb[slot*10+:10]=cb[slot*10+:10]|cmd_row_bases[r*10+:10];
   end
  end
  // A wrapped plane must be a different physical row, not a nonadjacent
  // alias of its first slot. Adjacent aliases were checked above.
  for(integer p=0;p<3;p=p+1)if(cn[p*2] && cn[p*2+1] && cb[p*20+:10]==cb[p*20+10+:10])config_bad=1;
 end
 wire valid_groups=cmd_groups==1 || cmd_groups==2 || cmd_groups==4;
 wire valid_out_groups=cmd_out_groups==1 || cmd_out_groups==2 || cmd_out_groups==4;
 wire [11:0] output_span=cmd_pair && !cmd_rgb ? ({3'd0,cmd_pixels}<<1)-12'd1 : {3'd0,cmd_pixels};
 wire good=!config_bad && !layout_bad && cmd_pixels>=1 && cmd_pixels<=320 && cmd_rows>=1 && cmd_rows<=4 && valid_groups && valid_out_groups &&
  cmd_source_width>=1 && cmd_source_width<=1280 && cmd_source_origin<(cmd_nn_view ? cmd_source_width>>1:cmd_source_width) &&
  {1'b0,cmd_output_x}+output_span<={1'b0,cmd_source_width} &&
  (cmd_pair ? (!cmd_nn_view && (cmd_rgb ? (cmd_groups==1 && cmd_group==0 && cmd_source_origin<=cmd_output_x) :
   (cmd_rows<=2 && cmd_groups<=2 && {1'b0,cmd_group}<cmd_groups && (cmd_output_x==0 ? cmd_source_origin==0:cmd_source_origin<cmd_output_x)))) :
   ((!cmd_nn_view || (!cmd_mode_pw && !cmd_source_width[0])) && (cmd_mode_pw ? cmd_source_origin<=cmd_output_x :
    ({1'b0,cmd_group}<cmd_groups && cmd_groups==cmd_out_groups && (cmd_output_x==0 ? cmd_source_origin==0:
     cmd_source_origin<=((cmd_output_x-11'd1)>>(cmd_nn_view ? 1:0)))))));
 wire active=!reset && !cancel && !error;
 assign cmd_ready=active && !running && !done;assign done_valid=done && !reset && !cancel;assign busy=running||done;
 wire [1:0] group_q=(!paired && is_pw) ? phase[4:3] & (groups[1:0]-2'd1):selected_group;
 wire [13:0] normal_raw=is_pw ? {3'd0,ox}+{2'd0,batch,3'd0}+{11'd0,phase[2:0]} :
  warmup ? {3'd0,ox}-14'd1 : {3'd0,ox}+{2'd0,batch,3'd0}+{7'd0,(phase==0 ? 7'd0:phase-7'd1)};
 wire signed [14:0] pair_raw=$signed({4'd0,ox})+$signed({1'b0,next_index,1'b0})-(rgb ? 15'sd0:15'sd1);
 wire signed [14:0] raw_a=paired ? pair_raw : (!is_pw && warmup && ox==0) ? 15'sd0:$signed({1'b0,normal_raw});
 wire signed [14:0] raw_b=raw_a+(paired ? 15'sd1:15'sd0);
 wire [10:0] xa=raw_a<0 ? 11'd0 : raw_a>=$signed({4'd0,source_width}) ? source_width-11'd1:raw_a[10:0];
 wire [10:0] xb=raw_b<0 ? 11'd0 : raw_b>=$signed({4'd0,source_width}) ? source_width-11'd1:raw_b[10:0];
 wire [10:0] physical_a=nn_view ? xa>>1:xa,physical_b=nn_view ? xb>>1:xb;
 wire [10:0] la=physical_a-origin,lb=physical_b-origin;wire [1:0] ca=la[1:0],cc=lb[1:0];
 wire skip=paired ? (rgb ? next_index>=(({4'd0,pixels}+13'd1)>>1):next_index>{4'd0,pixels}) : !warmup && normal_raw>last_needed;
 wire reuse_packet=!paired && !skip && previous_valid && previous_x==physical_a && previous_group==group_q;
 wire [23:0] head_mask=q_ports[head];wire [23:0] available=have;
 assign out_valid=active && running && live[head] && (q_skip[head] || q_reuse[head] || (available & head_mask)==head_mask);
 assign out_index=q_index[head];assign out_warmup=!paired && !is_pw && q_index[head]==0;
 wire pop=out_valid && out_ready;
 wire [23:0] occupied=(live[0] ? q_ports[0]:24'd0)|(live[1] ? q_ports[1]:24'd0)|
  (live[2] ? q_ports[2]:24'd0)|(live[3] ? q_ports[3]:24'd0);
 wire [23:0] occupied_after_pop=(live[0] && !(pop && head==0) ? q_ports[0]:24'd0)|
  (live[1] && !(pop && head==1) ? q_ports[1]:24'd0)|
  (live[2] && !(pop && head==2) ? q_ports[2]:24'd0)|
  (live[3] && !(pop && head==3) ? q_ports[3]:24'd0);
 wire [23:0] needed_ports;wire [5:0] address_bad;
 wire start_packet=active && running && next_index<total && (!live[tail] || (pop && tail==head)) &&
  (reuse_packet || (needed_ports & occupied_after_pop)==0);
 wire issuing=active && running;
 assign port_locked=(held|pending)&{24{active}};
 wire [23:0] reserve_ports={24{start_packet && !reuse_packet && !(|address_bad)}} & needed_ports;
 assign mem_req_valid={24{issuing}} & (todo | reserve_ports) & (port_available|held);
 wire [23:0] fire=mem_req_valid&mem_req_ready;
 assign mem_rsp_ready=pending & {24{active}};
 wire [63:0] port_word[0:23],group_a[0:5],group_b[0:5];
 for(genvar p=0;p<24;p=p+1)begin:PORT
  assign mem_req_address[p*10+:10]=reserve_ports[p] ? prepared_address[p*10+:10]:request_address[p];
  assign mem_req_tag[p*16+:16]=16'd0;
  assign port_word[p]=payload[p];
 end
 for(genvar g=0;g<6;g=g+1)begin:GROUP
  wire [12:0] offset_a=groups==4 ? {2'd0,la[10:2],2'd0}:groups==2 ? {3'd0,la[10:2],1'b0}:{4'd0,la[10:2]};
  wire [12:0] offset_b=groups==4 ? {2'd0,lb[10:2],2'd0}:groups==2 ? {3'd0,lb[10:2],1'b0}:{4'd0,lb[10:2]};
  wire [13:0] addr_a={4'd0,bases[g*10+:10]}+{1'b0,offset_a}+{12'd0,group_q};
  wire [13:0] addr_b={4'd0,bases[g*10+:10]}+{1'b0,offset_b}+{12'd0,group_q};
  assign address_bad[g]=!skip && !reuse_packet && needed[g] && (addr_a>=1024 || addr_b>=1024 || physical_a<origin || physical_b<origin);
  reg [63:0] a_word,b_word;
  always @*begin
   a_word=0;b_word=0;
   for(integer c=0;c<4;c=c+1)begin
    a_word=a_word|({64{q_a[head]==2'(c)}} & port_word[(g/2)*8+c*2+g%2]);
    b_word=b_word|({64{q_b[head]==2'(c)}} & port_word[(g/2)*8+c*2+g%2]);
   end
  end
  assign group_a[g]=a_word;assign group_b[g]=b_word;
  for(genvar c=0;c<4;c=c+1)begin:COLUMN
   localparam integer P=(g/2)*8+c*2+g%2;
   assign needed_ports[P]=!skip && needed[g] && (ca==2'(c) || cc==2'(c));
   assign prepared_address[P*10+:10]=ca==2'(c) ? addr_a[9:0]:addr_b[9:0];
  end
 end
 for(genvar r=0;r<6;r=r+1)begin:ROW
  reg [63:0] a_word,b_word;
  always @*begin
   a_word=0;b_word=0;
   for(integer g=0;g<6;g=g+1)if((r>=3 || g%2==0) && (r!=5 || (g!=0 && g!=4)))begin
    a_word=a_word|({64{row_map[r*3+:3]==3'(g)}} & group_a[g]);
    b_word=b_word|({64{row_map[r*3+:3]==3'(g)}} & group_b[g]);
   end
  end
  assign out_data[r*64+:64]=q_skip[head] ? 64'd0:a_word;
  assign out_data[(r+6)*64+:64]=(r==5 || !paired || q_skip[head]) ? 64'd0:b_word;
 end
 wire [2:0] log_period=(cmd_groups==4 ? 3'd2:cmd_groups==2 ? 3'd1:3'd0)+(cmd_out_groups==4 ? 3'd2:cmd_out_groups==2 ? 3'd1:3'd0);
 wire [12:0] batches=({4'd0,cmd_pixels}+13'd7)>>3;
 wire [12:0] normal_total=cmd_mode_pw ? (batches<<(log_period+3'd3))+{10'd0,(cmd_pixels[2:0]-3'd1)} :
  batches*13'd9+{10'd0,(cmd_pixels[2:0]-3'd1)}+13'd1;
 integer nread;
 always @*begin nread=0;for(integer p=0;p<24;p=p+1)nread=nread+int'(fire[p]);end
 always @(posedge clk)begin
  if(reset || cancel)begin
   running<=0;done<=0;error<=0;paired<=0;rgb<=0;is_pw<=0;nn_view<=0;warmup<=0;
   pixels<=0;batch<=0;groups<=0;selected_group<=0;ox<=0;source_width<=0;origin<=0;bases<=0;row_map<=0;needed<=0;
   next_index<=0;total<=0;phase<=0;phase_mask<=0;last_needed<=0;head<=0;tail<=0;live<=0;q_skip<=0;q_reuse<=0;
   todo<=0;held<=0;pending<=0;have<=0;read_words<=0;previous_valid<=0;previous_x<=0;previous_group<=0;
   for(integer q=0;q<4;q=q+1)begin q_ports[q]<=0;q_a[q]<=0;q_b[q]<=0;q_index[q]<=0;end
   for(integer p=0;p<24;p=p+1)begin request_address[p]<=0;end
  end else if(active)begin
   if(done && done_ready)done<=0;
   if(cmd_valid && cmd_ready)begin
    running<=good;done<=!good;error<=!good;paired<=cmd_pair;rgb<=cmd_rgb;is_pw<=cmd_mode_pw;nn_view<=cmd_nn_view;
    pixels<=cmd_pixels;groups<=cmd_groups;selected_group<=cmd_group;batch<=0;phase<=0;warmup<=!cmd_mode_pw && !cmd_pair;
    ox<=cmd_output_x;source_width<=cmd_source_width;origin<=cmd_source_origin;bases<=cb;row_map<=cm;needed<=cn;
    next_index<=0;total<=cmd_pair ? (cmd_rgb ? (({4'd0,cmd_pixels}+13'd1)>>1)+13'd7:{4'd0,cmd_pixels}+13'd8):normal_total;
    phase_mask<=cmd_mode_pw ? 7'((8<<log_period)-1):7'd8;
    last_needed<={3'd0,cmd_output_x}+{5'd0,cmd_pixels}-(cmd_mode_pw ? 14'd1:14'd0);
    head<=0;tail<=0;live<=0;todo<=0;held<=0;pending<=0;have<=0;previous_valid<=0;read_words<=0;
   end else read_words<=read_words+32'(nread);
   for(integer p=0;p<24;p=p+1)begin
    if(mem_rsp_valid[p])begin
     if(!pending[p] || mem_rsp_was_write[p])error<=1;
     if(mem_rsp_ready[p])begin pending[p]<=0;have[p]<=1;payload[p]<=mem_rsp_data[p*64+:64];end
    end
    if(pop && head_mask[p])have[p]<=0;
    if(mem_req_valid[p] && !mem_req_ready[p])held[p]<=1;
    if(fire[p])begin held[p]<=0;pending[p]<=1;have[p]<=0;todo[p]<=0;end
    // Live packet port masks are disjoint. Reserve each physical port once;
    // a blocked port cannot stop requests belonging to another packet.
    if(start_packet && !reuse_packet && needed_ports[p])begin
     todo[p]<=!fire[p];have[p]<=0;request_address[p]<=prepared_address[p*10+:10];
    end
   end
   if(pop)begin live[head]<=0;head<=head+2'd1;end
   if(start_packet)begin
    live[tail]<=1;q_ports[tail]<=needed_ports;q_a[tail]<=ca;q_b[tail]<=cc;q_index[tail]<=next_index;
    q_skip[tail]<=skip;q_reuse[tail]<=reuse_packet;
   end
   if(start_packet)begin
    if(|address_bad)error<=1;
    tail<=tail+2'd1;next_index<=next_index+13'd1;
    if(!skip)begin previous_valid<=1;previous_x<=physical_a;previous_group<=group_q;end
    if(warmup)warmup<=0;else if(phase==phase_mask)begin phase<=0;batch<=batch+9'd1;end else phase<=phase+7'd1;
   end
   if(running && next_index==total && live==0 && todo==0 && pending==0 && held==0)begin running<=0;done<=1;end
  end
 end
endmodule
