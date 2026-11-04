// Natural-order production columns, coupled to row_pair_arena6 A ports.
// One request per needed physical bank, with shared responses for clamped
// or NN-duplicated logical rows. A fall-through skid slot breaks arithmetic
// READY feedback while leaving unstalled RAM responses on the direct path.
module fixed_source_columns6(
 input wire clk,reset,cancel,
 input wire cmd_valid,output wire cmd_ready,input wire cmd_pw,cmd_stem,cmd_stride2,cmd_nn,cmd_stride4,cmd_nn4,
 input wire [8:0] cmd_pixels,input wire [2:0] cmd_rows,cmd_groups,input wire [1:0] cmd_group,
 input wire [10:0] cmd_output_x,cmd_source_width,cmd_source_origin,input wire cmd_storage_phase,
 input wire [17:0] cmd_row_banks,input wire [59:0] cmd_row_bases,
 output wire out_valid,input wire out_ready,output wire [383:0] out_data,
 output wire [8:0] out_column,output wire [1:0] out_part,output wire [2:0] out_parts,
 output wire done_valid,input wire done_ready,output wire busy,output reg error,
 output wire [5:0] read_valid,input wire [5:0] read_ready,output wire [59:0] read_address,
 input wire [5:0] read_response,output wire [5:0] read_response_ready,input wire [767:0] read_data
);
 // The production descriptor owner holds all configuration until completion.
 // Only request/response position is local state; do not copy the descriptor.
 reg running,done,pending,half_q;
 reg [8:0] next_column,column_q;reg [1:0] next_part,part_q;
 reg signed [11:0] raw_cursor;reg half_cursor;
 reg [11:0] offset_cursor;
 wire pw=cmd_pw,nn=cmd_nn || cmd_nn4,storage_phase=cmd_storage_phase;
 wire [1:0] nn_shift=cmd_nn4 ? 2'd2:cmd_nn ? 2'd1:2'd0;
 wire [8:0] columns;
 wire [1:0] group_q=cmd_group;
 wire [2:0] groups=cmd_groups;wire [2:0] parts;
 wire [3:0] rows;
 wire [10:0] ox=cmd_output_x,width=cmd_source_width,origin=cmd_source_origin;
 wire [17:0] row_map=cmd_row_banks;reg [59:0] bases;reg [5:0] needed;
 // Stride4 descriptors own one output row and three source rows. Stream
 // contiguous x values so the existing trained 3x3 MAC window is unchanged.
 wire [3:0] source_rows=cmd_pw ? {1'b0,cmd_rows}:cmd_stride4 ? 4'd3:cmd_stride2 ? ({1'b0,cmd_rows}<<1)+4'd1:{1'b0,cmd_rows}+4'd2;
 wire [10:0] input_columns=cmd_pw ? {2'd0,cmd_pixels}:cmd_stride4 ? ({2'd0,cmd_pixels}<<2)-11'd1:cmd_stride2 ? ({2'd0,cmd_pixels}<<1)+11'd1:{2'd0,cmd_pixels}+11'd2;
 wire [2:0] input_parts=cmd_stem ? 3'd3:cmd_pw ? cmd_groups:3'd1;
 reg [59:0] mapped_bases;reg [5:0] mapped_needed;reg config_bad;
 assign columns=input_columns[8:0];assign parts=input_parts;assign rows=source_rows;
 // Resolve the physical bank descriptor once at command acceptance. The
 // datapath starts on the following cycle, so this adds no request latency.
 // Configuration and alias checks remain on acceptance, not on every beat.
 always @(posedge clk)if(cmd_valid && cmd_ready)begin
  bases<=mapped_bases;needed<=mapped_needed;
 end
 always @*begin
  // Legal aliases have identical bases; OR selection removes priority muxes.
  // Conflicting aliases still reject the command before any RAM request.
  mapped_bases=0;mapped_needed=0;config_bad=source_rows>6;
  for(integer r=0;r<6;r=r+1)if(4'(r)<source_rows)begin
   if(cmd_row_banks[r*3+:3]>=6)config_bad=1;
   for(integer s=0;s<r;s=s+1)if(cmd_row_banks[r*3+:3]==cmd_row_banks[s*3+:3] && cmd_row_bases[r*10+:10]!=cmd_row_bases[s*10+:10])config_bad=1;
   for(integer b=0;b<6;b=b+1)if(cmd_row_banks[r*3+:3]==3'(b))begin mapped_bases[b*10+:10]=mapped_bases[b*10+:10] | cmd_row_bases[r*10+:10];mapped_needed[b]=1;end
  end
 end
 wire [11:0] span=cmd_stride4 ? ({3'd0,cmd_pixels}<<2)-12'd3:cmd_stride2 ? ({3'd0,cmd_pixels}<<1)-12'd1:{3'd0,cmd_pixels};
 wire good=!config_bad && cmd_pixels>=1 && cmd_pixels<=320 && cmd_rows>=1 && cmd_rows<=4 &&
  (cmd_groups==1 || cmd_groups==2 || cmd_groups==4) && cmd_source_width>=1 && cmd_source_width<=1280 &&
  {1'b0,cmd_output_x}+span<={1'b0,cmd_source_width} &&
  cmd_source_origin<(cmd_source_width>>nn_shift) && !(cmd_nn && cmd_nn4) && !(cmd_stride2 && cmd_stride4) &&
  (cmd_pw ? (!cmd_stem && !cmd_stride2 && !cmd_stride4 && !nn && cmd_source_origin<=cmd_output_x):
   ({1'b0,cmd_group}<cmd_groups && (cmd_output_x==0 ? cmd_source_origin==0:cmd_source_origin<=((cmd_output_x-11'd1)>>nn_shift)))) &&
  (!cmd_stem || (!cmd_stride2 && !cmd_stride4 && !nn && cmd_groups==1 && cmd_group==0)) &&
  (!cmd_stride2 || (!nn && cmd_rows<=2 && cmd_pixels<=160)) &&
  (!cmd_stride4 || (!nn && cmd_rows==1 && cmd_pixels<=128 && cmd_output_x[1:0]==0)) &&
  (!cmd_nn || !cmd_source_width[0]) && (!cmd_nn4 || cmd_source_width[1:0]==0);
 wire active=!reset && !cancel && !error;
 assign cmd_ready=active && !running && !done;assign busy=running || done;
 // RAM responses may pass directly, or enter one skid slot on backpressure.
 // RAM acceptance depends only on registered slot occupancy, not MAC READY.
 reg held_valid;
 reg [394:0] held_payload;
 wire memory_valid=active && pending && (read_response & needed)==needed;
 wire pop,issue;
 wire completing=running && next_column==columns &&
  (held_valid ? (!pending && out_ready):(!pending || (memory_valid && out_ready)));
 assign done_valid=active && (done || completing);
 // Compute the starting coordinate once per accepted command. During a
 // scan use bounded cursors; remove the serial coordinate/clamp/subtract/
 // multiply/add chain from the per-beat READY and watchdog paths.
 wire [10:0] first_x=pw || ox==0 ? ox:ox-11'd1;
 wire [10:0] first_sx=first_x>>nn_shift;
 wire [11:0] first_local={1'b0,first_sx}-{1'b0,origin}+{11'd0,storage_phase};
 wire [13:0] first_offset=groups==4 ? {1'b0,first_local[11:1],2'd0}:
  groups==2 ? {2'd0,first_local[11:1],1'b0}:{3'd0,first_local[11:1]};
 wire [1:0] first_group=pw ? 2'd0:group_q;
 wire final_part={1'b0,next_part}+3'd1==parts;
 wire increment_pixel=raw_cursor>=0 && raw_cursor<$signed({1'b0,width})-12'sd1 &&
  (!nn || (cmd_nn4 ? &raw_cursor[1:0]:raw_cursor[0]));
 wire next_pair=increment_pixel && half_cursor;
 wire [11:0] backstep=groups==4 ? 12'hffd:groups==2 ? 12'hfff:12'd0;
 wire [11:0] advance_words=pw ? (!final_part || next_pair ? 12'd1:backstep):
  (final_part && next_pair ? {9'd0,groups}:12'd0);
 wire [14:0] first_word={1'b0,first_offset}+{13'd0,first_group};
 always @(posedge clk)begin
  if(cmd_valid && cmd_ready)offset_cursor<=first_word[11:0];
  else if(issue)offset_cursor<=offset_cursor+advance_words;
 end
 wire [5:0] address_bad;
 for(genvar b=0;b<6;b=b+1)begin:ADDRESS
  wire [12:0] address={3'd0,bases[b*10+:10]}+{1'b0,offset_cursor};
  assign read_address[b*10+:10]=address[9:0];
  assign address_bad[b]=needed[b] && |address[12:10];
 end
 assign out_valid=active && (held_valid || memory_valid);
 assign out_parts=parts;
 assign pop=memory_valid && !held_valid;
 wire want=active && running && next_column<columns && (!pending || pop);
 assign issue=want && !(|address_bad) && (read_ready & needed)==needed;
 assign read_valid=needed & {6{issue}};assign read_response_ready=needed & {6{pop}};
 // Select the 64-bit half before the six-way row selector.
 wire [383:0] halves;
 for(genvar b=0;b<6;b=b+1)assign halves[b*64+:64]=half_q ? read_data[b*128+64+:64]:read_data[b*128+:64];
 wire [383:0] selected_columns;
 for(genvar r=0;r<6;r=r+1)begin:ROW
  reg [63:0] selected;
  always @*begin
   selected=0;
   for(integer b=0;b<6;b=b+1)selected=selected|({64{row_map[r*3+:3]==3'(b) && 4'(r)<rows}} & halves[b*64+:64]);
  end
  assign selected_columns[r*64+:64]=selected;
 end
 wire [394:0] direct_payload={column_q,part_q,selected_columns};
 assign {out_column,out_part,out_data}=held_valid ? held_payload:direct_payload;
 always @(posedge clk)begin
  if(reset || cancel)held_valid<=0;
  else if(active)begin
   if(held_valid)begin
    if(out_ready)held_valid<=0;
   end else if(memory_valid && !out_ready)held_valid<=1;
  end
  // Capture every admitted RAM response. When it passes through, the saved
  // value is unobserved; avoiding out_ready here removes its 395-bit CE load.
  if(pop)held_payload<=direct_payload;
 end
 // synthesis translate_off
 always @(posedge clk)if(active)begin
  if(held_valid && pop)$fatal(1,"SOURCE_SKID_OVERWRITE");
  if(cmd_valid && cmd_ready && held_valid)$fatal(1,"SOURCE_SKID_NOT_DRAINED");
 end
 // synthesis translate_on
 always @(posedge clk)begin
  if(reset || cancel)begin
   running<=0;done<=0;pending<=0;half_q<=0;error<=0;
   next_column<=0;column_q<=0;next_part<=0;part_q<=0;raw_cursor<=0;half_cursor<=0;
  end else if(active)begin
   if(cmd_valid && cmd_ready)begin
    running<=good;done<=!good;error<=!good || cmd_rows[2];
    next_column<=0;next_part<=0;pending<=0;
    raw_cursor<=$signed({1'b0,ox})-(pw ? 12'sd0:12'sd1);half_cursor<=first_local[0];
   end
   if(pop)pending<=0;
   if(want && |address_bad)error<=1;
   if(issue)begin
    pending<=1;column_q<=next_column;part_q<=next_part;half_q<=half_cursor;
    if(final_part)begin
     next_part<=0;next_column<=next_column+9'd1;raw_cursor<=raw_cursor+12'sd1;
     if(increment_pixel)half_cursor<=!half_cursor;
    end else next_part<=next_part+2'd1;
   end
   if(completing)begin running<=0;done<=1;end
   if(done_valid && done_ready)done<=0;
  end
 end
 // synthesis translate_off
 localparam integer CFG_W=6+9+3+3+2+33+1+18+60;
 wire [CFG_W-1:0] config_view={cmd_pw,cmd_stem,cmd_stride2,cmd_nn,cmd_stride4,cmd_nn4,cmd_pixels,cmd_rows,cmd_groups,cmd_group,
  cmd_output_x,cmd_source_width,cmd_source_origin,cmd_storage_phase,cmd_row_banks,cmd_row_bases};
 reg [CFG_W-1:0] held_config;
 always @(posedge clk)begin
  if(cmd_valid && cmd_ready)held_config<=config_view;
  if(!reset && !cancel && running && config_view!==held_config)$fatal(1,"FIXED_SOURCE_OWNER_CONFIG_CHANGED");
  if(!reset && !cancel && running && {bases,needed}!=={mapped_bases,mapped_needed})
   $fatal(1,"FIXED_SOURCE_MAPPED_DESCRIPTOR_CHANGED");
 end
 // synthesis translate_on
endmodule
