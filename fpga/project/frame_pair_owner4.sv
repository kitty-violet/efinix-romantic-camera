// Four paired style slots (native uses the original first three), with a retained display front. One reader owns a
// slot until reader_done (last accepted source pixel, all DDR reads drained).
// Compact ranks are a permutation of 0,1,2,3; no wrapping serial comparison.
module frame_pair_owner4(
 input wire clk,reset,bypass_compute,capture_native,
 input wire capture_valid,output wire capture_ready,input wire [15:0] capture_frame_id,
 output wire [1:0] capture_bank,output wire [31:0] capture_address,
 input wire capture_done,capture_failed,
 output wire compute_valid,input wire compute_ready,
 output wire [15:0] compute_frame_id,output wire [31:0] compute_input_address,compute_output_address,
 input wire compute_done,compute_failed,compute_discarded,
 output wire display_valid,input wire display_ready,
 output wire [15:0] display_frame_id,output wire [31:0] display_input_address,display_output_address,
 input wire display_done,
 output reg error,output wire [11:0] states,
 output reg [31:0] committed_frames,output wire [31:0] dropped_results,
 output wire [31:0] repeated_reads,
 output wire front_valid,output wire [1:0] front_bank
,
 output wire [143:0] compat_state,
 output wire [31:0] slot3_observer,
 output wire work_pending,
 output wire [127:0] service_native_state
);
 localparam [2:0] FREE=0,CAPTURE=1,CAPTURED=2,COMPUTE=3,RESULT=4,DISPLAY=5,FRONT=6;
 reg [2:0] state[0:3];reg [15:0] frame[0:3];reg [1:0] rank[0:3];
 reg[15:0] dropped_count,repeated_count;
 assign dropped_results={16'd0,dropped_count};
 assign repeated_reads={16'd0,repeated_count};
 reg capturing,computing,displaying,c_offer,d_offer,has_front;
 reg [1:0] cap_slot,comp_slot,disp_slot,c_candidate,d_candidate,front_slot;
 reg cap_found,comp_found,disp_found;reg [1:0] free_choice,comp_choice,disp_choice;
 always @*begin
  cap_found=0;comp_found=0;disp_found=0;free_choice=0;comp_choice=0;disp_choice=0;
  for(integer b=0;b<4;b=b+1)begin
   if(state[b]==FREE && !cap_found && (b!=3 || !capture_native))begin cap_found=1;free_choice=2'(b);end
   if(state[b]==CAPTURED && (!comp_found || rank[b]>rank[comp_choice]))begin comp_found=1;comp_choice=2'(b);end
   if(state[b]==RESULT && (!disp_found || rank[b]<rank[disp_choice]))begin disp_found=1;disp_choice=2'(b);end
  end
  if(!disp_found && has_front && state[front_slot]==FRONT)begin disp_found=1;disp_choice=front_slot;end
 end
 // Keep original three paired DDR allocations. New pair is in the unused
 // 24..32MiB region, below the encoder workspace at32MiB.
 // Slot3 input has bit24 set: integration MUST widen retained input_region
 // and its physical DDR relocation mux to three bits; silently truncating
 // it would write/read slot0 output at12MiB. See address proof/manifest.
 function automatic [31:0] input_base(input [1:0] bank);
  input_base=bank==3 ? 32'h01c00000 : {8'd0,bank,22'd0};
 endfunction
 function automatic [31:0] output_base(input [1:0] bank);
  output_base=bank==3 ? 32'h01800000 : 32'h00c00000+{8'd0,bank,22'd0};
 endfunction
 assign capture_ready=!reset && !error && !capturing && cap_found;
 assign capture_bank=free_choice;
 assign capture_address=input_base(free_choice);
 assign compute_valid=!reset && !error && c_offer;
 assign compute_frame_id=frame[c_candidate];
 assign compute_input_address=input_base(c_candidate);
 assign compute_output_address=output_base(c_candidate);
 assign display_valid=!reset && !error && d_offer;
 assign display_frame_id=frame[d_candidate];
 assign display_input_address=input_base(d_candidate);
 assign display_output_address=output_base(d_candidate);
 assign front_valid=has_front;assign front_bank=front_slot;
 for(genvar b=0;b<4;b=b+1)assign states[b*3+:3]=state[b];
 wire cap_fire=capture_valid && capture_ready,comp_fire=compute_valid && compute_ready,disp_fire=display_valid && display_ready;
 reg [1:0] drop_count;
 always @*begin
  drop_count=0;
  for(integer b=0;b<4;b=b+1)if(state[b]==RESULT && 2'(b)!=d_candidate && rank[b]>rank[d_candidate])drop_count=drop_count+1'b1;
 end
 always @(posedge clk)begin
  if(reset)begin
   capturing<=0;computing<=0;displaying<=0;c_offer<=0;d_offer<=0;has_front<=0;
   cap_slot<=0;comp_slot<=0;disp_slot<=0;c_candidate<=0;d_candidate<=0;front_slot<=0;
   error<=0;committed_frames<=0;dropped_count<=0;repeated_count<=0;
   for(integer b=0;b<4;b=b+1)begin state[b]<=FREE;frame[b]<=0;rank[b]<=2'(b);end
  end else if(!error)begin
   if(!computing && !c_offer && comp_found)begin c_candidate<=comp_choice;c_offer<=1;end
   if(!displaying && !d_offer && disp_found)begin d_candidate<=disp_choice;d_offer<=1;end
   if(cap_fire)begin
    capturing<=1;cap_slot<=free_choice;state[free_choice]<=CAPTURE;frame[free_choice]<=capture_frame_id;
    for(integer b=0;b<4;b=b+1)begin
     if(2'(b)==free_choice)rank[b]<=0;
     else if(rank[b]<rank[free_choice])rank[b]<=rank[b]+1'b1;
    end
   end
   if(capture_done || capture_failed)begin
    if(!capturing || state[cap_slot]!=CAPTURE)error<=1;
    else begin
     capturing<=0;state[cap_slot]<=capture_failed ? FREE:(bypass_compute ? RESULT:CAPTURED);
     if(!capture_failed && bypass_compute)committed_frames<=committed_frames+1'b1;
    end
   end
   if(comp_fire)begin computing<=1;c_offer<=0;comp_slot<=c_candidate;state[c_candidate]<=COMPUTE;end
   if(compute_done || compute_failed || compute_discarded)begin
    if(!computing || state[comp_slot]!=COMPUTE)error<=1;
    else begin
     computing<=0;
     // Failures retain priority; a drained discard never publishes a result.
     state[comp_slot]<=compute_failed ? FREE:(compute_discarded ? CAPTURED:RESULT);
     if(!compute_failed && !compute_discarded)committed_frames<=committed_frames+1'b1;
    end
   end
   if(disp_fire)begin
    displaying<=1;d_offer<=0;disp_slot<=d_candidate;state[d_candidate]<=DISPLAY;
    if(has_front && front_slot!=d_candidate)state[front_slot]<=FREE;
    if(has_front && front_slot==d_candidate)repeated_count<=repeated_count+1'b1;
    has_front<=1;front_slot<=d_candidate;
    for(integer b=0;b<4;b=b+1)
     if(state[b]==RESULT && 2'(b)!=d_candidate && rank[b]>rank[d_candidate])state[b]<=FREE;
    dropped_count<=dropped_count+{14'd0,drop_count};
   end
   if(display_done)begin
    if(!displaying || state[disp_slot]!=DISPLAY)error<=1;
    else begin displaying<=0;state[disp_slot]<=FRONT;end
   end
  end
 end
 assign service_native_state={frame[2][9], frame[2][8], frame[2][7], frame[2][6], frame[2][5], frame[2][4], frame[2][3], frame[2][2], frame[2][1], frame[2][15], frame[2][14], frame[2][13], frame[2][12], frame[2][11], frame[2][10], frame[2][0], frame[1][9], frame[1][8], frame[1][7], frame[1][6], frame[1][5], frame[1][4], frame[1][3], frame[1][2], frame[1][1], frame[1][15], frame[1][14], frame[1][13], frame[1][12], frame[1][11], frame[1][10], frame[1][0], frame[0][9], frame[0][8], frame[0][7], frame[0][6], frame[0][5], frame[0][4], frame[0][3], frame[0][2], frame[0][1], frame[0][15], frame[0][14], frame[0][13], frame[0][12], frame[0][11], frame[0][10], frame[0][0], capturing, c_offer, d_candidate[1], d_candidate[0], c_candidate[1], c_candidate[0], state[2][2], state[2][1], state[2][0], state[1][2], state[1][1], state[1][0], state[0][2], state[0][1], state[0][0], dropped_results[9], dropped_results[8], dropped_results[7], dropped_results[6], dropped_results[5], dropped_results[4], dropped_results[3], dropped_results[31], dropped_results[30], dropped_results[2], dropped_results[29], dropped_results[28], dropped_results[27], dropped_results[26], dropped_results[25], dropped_results[24], dropped_results[23], dropped_results[22], dropped_results[21], dropped_results[20], dropped_results[1], dropped_results[19], dropped_results[18], dropped_results[17], dropped_results[16], dropped_results[15], dropped_results[14], dropped_results[13], dropped_results[12], dropped_results[11], dropped_results[10], dropped_results[0], committed_frames[9], committed_frames[8], committed_frames[7], committed_frames[6], committed_frames[5], committed_frames[4], committed_frames[3], committed_frames[31], committed_frames[30], committed_frames[2], committed_frames[29], committed_frames[28], committed_frames[27], committed_frames[26], committed_frames[25], committed_frames[24], committed_frames[23], committed_frames[22], committed_frames[21], committed_frames[20], committed_frames[1], committed_frames[19], committed_frames[18], committed_frames[17], committed_frames[16], committed_frames[15], committed_frames[14], committed_frames[13], committed_frames[12], committed_frames[11], committed_frames[10], committed_frames[0], error};
assign compat_state={d_candidate[1],c_candidate[1],d_candidate[0],c_candidate[0],rank[2][1],rank[2][0],rank[1][1],rank[1][0],rank[0][1],rank[0][0],frame[2][15],frame[2][14],frame[2][13],frame[2][12],frame[2][11],frame[2][10],frame[2][9],frame[2][8],frame[2][7],frame[2][6],frame[2][5],frame[2][4],frame[2][3],frame[2][2],frame[2][1],frame[2][0],frame[1][15],frame[1][14],frame[1][13],frame[1][12],frame[1][11],frame[1][10],frame[1][9],frame[1][8],frame[1][7],frame[1][6],frame[1][5],frame[1][4],frame[1][3],frame[1][2],frame[1][1],frame[1][0],frame[0][15],frame[0][14],frame[0][13],frame[0][12],frame[0][11],frame[0][10],frame[0][9],frame[0][8],frame[0][7],frame[0][6],frame[0][5],frame[0][4],frame[0][3],frame[0][2],frame[0][1],frame[0][0],disp_slot[1],comp_slot[1],cap_slot[1],capturing,state[0][2],state[0][1],state[0][0],state[1][2],state[1][1],state[1][0],state[2][2],state[2][1],repeated_reads[15],repeated_reads[14],repeated_reads[13],repeated_reads[12],repeated_reads[11],repeated_reads[10],repeated_reads[9],repeated_reads[8],repeated_reads[7],repeated_reads[6],repeated_reads[5],repeated_reads[4],repeated_reads[3],repeated_reads[2],repeated_reads[1],dropped_results[15],dropped_results[14],committed_frames[25],repeated_reads[0],error,front_slot[0],comp_slot[0],has_front,d_offer,c_offer,displaying,computing,state[2][0],dropped_results[0],cap_slot[0],dropped_results[1],dropped_results[9],dropped_results[8],dropped_results[7],committed_frames[2],committed_frames[5],committed_frames[10],dropped_results[11],dropped_results[10],committed_frames[1],committed_frames[6],committed_frames[15],dropped_results[13],dropped_results[12],committed_frames[0],committed_frames[7],committed_frames[20],dropped_results[2],committed_frames[11],committed_frames[3],committed_frames[12],committed_frames[16],committed_frames[4],committed_frames[17],committed_frames[21],committed_frames[22],dropped_results[3],dropped_results[4],committed_frames[8],committed_frames[13],committed_frames[9],committed_frames[18],committed_frames[23],committed_frames[26],committed_frames[27],committed_frames[28],committed_frames[29],committed_frames[30],committed_frames[31],dropped_results[5],committed_frames[14],committed_frames[24],committed_frames[19],dropped_results[6]};
assign work_pending=capturing || computing || c_offer || state[0]==CAPTURED || state[1]==CAPTURED || state[2]==CAPTURED || state[3]==CAPTURED;
assign slot3_observer={8'hf4,2'd0,rank[3],frame[3],1'b0,state[3]};
endmodule
