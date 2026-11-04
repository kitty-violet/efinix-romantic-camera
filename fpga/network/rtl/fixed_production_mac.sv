// Natural production coordinates -> real SRAM -> fixed MAC -> partial sums.
// Exactly one output-channel group per command; acknowledged writer, coefficient
// image service and fused residual are connected by the outer production owner.
module fixed_production_mac #(parameter integer TAIL_SHIFT_048=0)(
 input wire clk,reset,cancel,
 input wire weight_begin,output wire weight_begin_ready,input wire [2:0] weight_parts,
 input wire weight_valid,output wire weight_ready,input wire [5:0] weight_index,
 input wire [63:0] weight_data,input wire weight_last,
 input wire cmd_valid,output wire cmd_ready,input wire cmd_pw,cmd_stem,cmd_stride2,cmd_nn,cmd_stride4,cmd_nn4,
 input wire [8:0] cmd_pixels,input wire [2:0] cmd_rows,cmd_groups,input wire [1:0] cmd_group,
 input wire [10:0] cmd_output_x,cmd_source_width,cmd_source_origin,input wire cmd_storage_phase,
 input wire [255:0] bias,
 input wire [17:0] cmd_row_banks,input wire [59:0] cmd_row_bases,
 output wire out_valid,input wire out_ready,output wire [767:0] out_sums,output wire [8:0] out_x,
 output wire done_valid,input wire done_ready,output wire busy,error,
 output wire [5:0] read_valid,input wire [5:0] read_ready,output wire [59:0] read_address,
 input wire [5:0] read_response,output wire [5:0] read_response_ready,input wire [767:0] read_data,
 output reg [12:0] accepted_vectors,output reg [8:0] results
);
 localparam IDLE=0,RUN=1,DRAIN=2,DONE=3;
 reg [1:0] state;reg fault;
 wire stem=cmd_stem,pw=cmd_pw,stride2=cmd_stride2;wire [8:0] pixels=cmd_pixels;
 wire [575:0] weights;wire weights_ready,werror,perror,rerror;
 wire [2:0] parts;reg [2:0] running_parts;wire [1:0] weight_phase;
 wire active=!reset && !cancel && !error;
 assign error=fault || werror || perror || rerror;assign busy=state!=IDLE;
 wire srcready,srcvalid,srctake,srcdone;wire [383:0] column_data;wire [8:0] column;wire [1:0] phase;wire [2:0] source_parts;
 assign cmd_ready=active && state==IDLE && srcready && weights_ready && !weight_begin && !weight_valid;
 wire start=cmd_valid && cmd_ready;
 wire correct_parts=parts==(cmd_stem ? 3'd3:cmd_pw ? cmd_groups:3'd1);
 fixed_source_columns6 source(.cmd_valid(start && correct_parts),.cmd_ready(srcready),
  .out_valid(srcvalid),.out_ready(srctake),.out_data(column_data),.out_column(column),.out_part(phase),.out_parts(source_parts),
  .done_valid(srcdone),.done_ready(1'b1),.busy(),.error(rerror),.*);
 wire cv=active && state==RUN && srcvalid;wire cr,cov,pr;wire [671:0] dots;
 wire step;wire consume=cv && cr;assign srctake=active && state==RUN && cr;
 fixed_ring_weights9 #(.SKEW_TERMS(1),.TAIL_SHIFT_048(TAIL_SHIFT_048),.ACTIVATE_INTERLOCKED(1)) cache(.clk(clk),.reset(reset),.cancel(cancel),.idle(state==IDLE || state==DONE),.activate(start),
  .begin_valid(weight_begin),.begin_ready(weight_begin_ready),.begin_parts(weight_parts),
  .load_valid(weight_valid),.load_ready(weight_ready),.load_index(weight_index),.load_data(weight_data),.load_last(weight_last),
  .step(step),.ready(weights_ready),.weights(weights),.error(werror),.parts(parts),.phase(weight_phase));
 // RGB stem stores only two preceding 24-bit columns in each logical row.
 // Three input-channel phases share this fixed spatial window.
 reg [23:0] previous[0:4],older[0:4];
 always @(posedge clk)if(consume && stem && phase==2)begin
  for(integer r=0;r<5;r=r+1)begin older[r]<=previous[r];previous[r]<=column_data[r*64+:24];end
 end
 wire [287:0] activation;
 assign activation[216+:72]=0;
 for(genvar r=0;r<3;r=r+1)for(genvar t=0;t<9;t=t+1)begin:A
  wire [23:0] rgb=t%3==0 ? older[r+t/3]:t%3==1 ? previous[r+t/3]:column_data[(r+t/3)*64+:24];
  wire [7:0] channel=phase==0 ? rgb[7:0]:phase==1 ? rgb[15:8]:rgb[23:16];
  if(t==8)assign activation[(r*9+t)*8+:8]=stem ? channel:8'd0;
  else assign activation[(r*9+t)*8+:8]=stem ? channel:column_data[r*64+t*8+:8];
 end
 wire drain=state==DRAIN;
 wire warm=!pw && (column<2 || (cmd_stride4 ? column[1:0]!=2'd2:(stride2 && column[0])));
 fixed_window_dot8x32 #(.WEIGHT_PRE_SKEWED(1),.SHARED_COLUMN_ACTIVATION(1),.TAIL_SHIFT_048(TAIL_SHIFT_048)) arithmetic(.clk(clk),.reset(reset || cancel || start),.depthwise(!pw && !stem),
  .drain(drain),.in_warmup(warm),.in_valid(cv),.in_ready(cr),.columns(column_data),.activation(activation),
  .weight(weights),.out_valid(cov),.out_ready(pr),.dots(dots));
 reg [8:0] tags[0:8];reg [1:0] phases[0:8];
 assign step=consume || (drain && (!cov || pr));
 always @(posedge clk)if(step)begin
  tags[0]<=pw ? column:cmd_stride4 ? (column-9'd2)>>2:stride2 ? (column-9'd2)>>1:column-9'd2;phases[0]<=phase;
  for(integer n=1;n<9;n=n+1)begin tags[n]<=tags[n-1];phases[n]<=phases[n-1];end
 end
 wire [671:0] selected_dots=cmd_stride4 ? {504'd0,dots[0+:168]}:stride2 ? {336'd0,dots[336+:168],dots[0+:168]}:dots;
 wire pv;
 fixed_partial32 #(.DOT_WIDTH(19)) partial(.clk(clk),.reset(reset || start),.cancel(cancel),.in_valid(cov),.in_ready(pr),
  .in_first(phases[8]==0),.in_last({1'b0,phases[8]}+3'd1==running_parts),.in_parts(running_parts),.in_x(tags[8]),.in_dots(selected_dots),.bias(bias),
  .out_valid(pv),.out_ready(out_ready && active),.out_sums(out_sums),.out_x(out_x),.busy(),.error(perror));
 assign out_valid=active && pv;
 assign done_valid=active && state==DONE;
 always @(posedge clk)begin
  if(reset || cancel)begin state<=IDLE;fault<=0;accepted_vectors<=0;results<=0;running_parts<=0;end
  else if(!error)begin
   if(start)begin
    if(!correct_parts)fault<=1;
    else begin state<=RUN;running_parts<=parts;accepted_vectors<=0;results<=0;end
   end
   if(consume)begin accepted_vectors<=accepted_vectors+13'd1;if(weight_phase!=phase || running_parts!=source_parts)fault<=1;end
   if(state==RUN && srcdone)state<=DRAIN;
   if(out_valid && out_ready)begin
    if(out_x!=results)fault<=1;results<=results+9'd1;
    if(results+9'd1==pixels)state<=DONE;
   end
   if(done_valid && done_ready)state<=IDLE;
  end
 end
endmodule
