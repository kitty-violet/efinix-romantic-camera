// Nine fixed kernel positions x eight output channels. Each bit is a
// depth8 SRL: at most four active phases plus four incoming phases.
// Per-term load counts retarget the active tap as new words shift in.
// This is two logical groups in the SAME576 SRLs, not a duplicated cache.
// Nine64-bit words per phase enter through one real load port.
module fixed_ring_weights9 #(parameter integer SKEW_TERMS=0, TAIL_SHIFT_048=0, ACTIVATE_INTERLOCKED=0)(
 input wire clk,reset,cancel,idle,activate,
 input wire begin_valid,output wire begin_ready,input wire [2:0] begin_parts,
 input wire load_valid,output wire load_ready,input wire [5:0] load_index,
 input wire [63:0] load_data,input wire load_last,
 input wire step,output wire ready,output wire [575:0] weights,
 output reg error,output reg [2:0] parts,output reg [1:0] phase
);
 reg loading,loaded,staged,active_loaded;reg [2:0] active_parts;
 reg [2:0] incoming[0:8];reg [5:0] count;reg [3:0] term;reg [1:0] load_phase;
 // Q is natively inverted. Complement once on the shared 64-bit loader
 // instead of after all 576 SRL outputs. Complement INIT as well so even
 // unqualified payloads retain the original initial value.
 (* syn_keep="true" *) wire [63:0] inverted_load=~load_data;
 wire active=!reset && !cancel && !error;
 // A speculative next group may be abandoned by a new descriptor. Once
 // the MAC is idle, no remaining input consumes the old/staged taps.
 // Fixed MAC already excludes both loading VALID signals from activate.
 // Generic users retain the original explicit interlock.
 wire activate_guard=ACTIVATE_INTERLOCKED!=0 || !activate;
 assign begin_ready=active && !loading && (!staged || idle) && activate_guard;
 assign load_ready=active && loading && !begin_valid && activate_guard;
 assign ready=active && loaded && !begin_valid;
 // synthesis translate_off
 always @(posedge clk)if(!reset && !cancel)begin
  if(loading && loaded)$fatal(1,"CACHE_LOADING_LOADED_OVERLAP");
  if(ACTIVATE_INTERLOCKED!=0 && activate && (begin_valid || load_valid))$fatal(1,"CACHE_ACTIVATE_INTERLOCK_BROKEN");
 end
 // synthesis translate_on
 wire take=load_valid && load_ready;
 wire last=(term==8 && {1'b0,load_phase}+3'd1==parts);
 wire [7:0] tail_supported;
 for(genvar c=0;c<8;c=c+1)begin:TAIL_GUARD
  // Evaluate the actual incoming byte, not a truncated/cache copy.
  assign tail_supported[c]=load_data[c*8+:8]==8'd0 ||
   load_data[c*8+:8]==8'd4 || load_data[c*8+:8]==8'd8;
 end
 wire good=load_index==count && load_last==last &&
  (TAIL_SHIFT_048==0 || term!=8 || (&tail_supported));
 wire rotate=step && active_loaded;
 for(genvar t=0;t<9;t=t+1)begin:TERM
  // Phase walks only on an accepted arithmetic wavefront step, including
  // drain. Select the historical phase here instead of duplicating delay SRLs.
  localparam integer MOD2=t%2,MOD3=t%3,MOD4=t%4;
  wire [2:0] delay_mod=active_parts==4 ? 3'(MOD4):active_parts==3 ? 3'(MOD3):active_parts==2 ? 3'(MOD2):3'd0;
  wire [3:0] phase_sum={2'd0,phase}+{1'b0,active_parts}-{1'b0,delay_mod};
  wire [2:0] historical_phase=phase_sum>={1'b0,active_parts} ? 3'(phase_sum-active_parts):phase_sum[2:0];
  wire [2:0] selected_phase=SKEW_TERMS!=0 ? historical_phase:{1'b0,phase};
  wire [2:0] address=3'd0-active_parts-incoming[t]+selected_phase;
  for(genvar c=0;c<8;c=c+1)begin:CHANNEL
   for(genvar b=0;b<8;b=b+1)begin:BIT
    wire q,value;
    wire write=take && good && term==t;
    EFX_SRL8 #(.INIT(8'hff)) storage(.D(inverted_load[c*8+b]),.A(address),.CLK(clk),.CE(write),.Q(q),.Q7());
    assign value=q;assign weights[(c*9+t)*8+b]=value;
   end
  end
 end
 always @(posedge clk)begin
  if(reset || cancel)begin loading<=0;loaded<=0;count<=0;term<=0;load_phase<=0;parts<=0;phase<=0;error<=0;staged<=0;active_loaded<=0;active_parts<=0;for(integer t=0;t<9;t=t+1)incoming[t]<=0;end
  else if(!error)begin
   if(step && !active_loaded)error<=1;
   if(begin_valid && begin_ready)begin
    count<=0;term<=0;load_phase<=0;loaded<=0;parts<=begin_parts;
    for(integer t=0;t<9;t=t+1)incoming[t]<=0;
    if(begin_parts<1 || begin_parts>4)begin error<=1;loading<=0;end else loading<=1;
   end else if(take)begin
    if(!good)begin error<=1;loading<=0;loaded<=0;end
    else begin
     count<=count+6'd1;incoming[term]<=incoming[term]+3'd1;
     if(term==8)begin term<=0;load_phase<=load_phase+2'd1;end else term<=term+4'd1;
     if(last)begin loaded<=1;staged<=1;loading<=0;end
    end
   end
   if(rotate)phase<={1'b0,phase}+3'd1==active_parts ? 2'd0:phase+2'd1;
   if(activate)begin
    if(!loaded || loading || step)error<=1;
    else begin active_loaded<=1;active_parts<=parts;phase<=0;staged<=0;for(integer t=0;t<9;t=t+1)incoming[t]<=0;end
   end
  end
 end
endmodule
