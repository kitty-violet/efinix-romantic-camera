// Version 3 feature layout: DDR and SRAM both store pixel-pair/C8-group words.
// RGB remains byte-packed. Aligned C32 edge reads occupy reserved padding.
// One depth-three 128-bit queue owns payload; no feature transpose is required.
module row_pair_dma(
 input wire clk,reset,cancel,
 input wire cmd_valid,output wire cmd_ready,input wire cmd_store,cmd_rgb,
 input wire [31:0] cmd_address,input wire [8:0] cmd_columns,input wire [2:0] cmd_groups,cmd_bank,
 input wire [9:0] cmd_base,input wire cmd_phase,
 output wire done_valid,input wire done_ready,output reg error,output wire busy,
 output wire [5:0] av,input wire [5:0] ar,output wire [11:0] ah,
 output wire [59:0] aa,output wire [767:0] ad,
 input wire [5:0] sv,output wire [5:0] sr,input wire [767:0] sd,input wire [5:0] sw,
 output wire [31:0] araddr,output wire [7:0] arlen,output wire arvalid,input wire arready,
 input wire [127:0] rdata,input wire [1:0] rresp,input wire rlast,rvalid,output wire rready,
 output wire [31:0] awaddr,output wire [7:0] awlen,output wire awvalid,input wire awready,
 output wire [127:0] wdata,output wire [15:0] wstrb,output wire wlast,wvalid,input wire wready,
 input wire [1:0] bresp,input wire bvalid,output wire bready,
 output reg [31:0] cycles,arena_words,axi_beats
);
 localparam [2:0] IDLE=0,AR=1,R=2,AW=3,W=4,B=5,DRAIN=6,DONE=7;
 reg [2:0] state,bank;reg store,rgb;reg [9:0] base;reg [8:0] columns;
 reg [31:0] address;reg [14:0] payload_bytes;reg phase;
 reg [10:0] request_count,issued;reg [10:0] remaining;reg [9:0] beat_index;
 reg [8:0] burst_left,burst_size;reg [3:0] prefix;reg first_read;
 reg pending,pending_second,arena_held;
 wire [127:0] feature_word;wire feature_valid;
 reg [1:0] feature_count;wire store_room,load_room;
 // RGB moves in even six-byte pairs and sixteen-byte AXI beats. A terminal
 // odd pixel uses two halfwords; its padding byte is excluded by WSTRB.
 // Eight fixed halfword banks eliminate variable 256-bit shift/append logic.
 reg [7:0] rgb_latest_slot;reg [3:0] rgb_head,rgb_tail;reg [5:0] rgb_count;
 wire [127:0] rgb_front,rgb_bank_window;
 // The live RGB/feature queue owns a stalled W payload; retain only its strobe.
 // Abort freezes both queues until the held beat is accepted, then zero-strobes drain.
 reg wheld,held_zero;reg [15:0] held_wstrb;
 wire active=state!=IDLE && state!=DONE && !reset;
 wire aborting=error || cancel;
 wire arena_response=active && !aborting && pending && sv[bank];
 wire [127:0] arena_data=sd[bank*128+:128];
 wire second=({1'b0,columns} > (rgb ? {issued[8:0],1'b0}: {1'b0,issued[10:2],1'b0})+11'd1);
 wire write_fire,arena_fire,read_fire;
 wire feature_pop=active && !rgb && !aborting && (store ? write_fire:arena_fire);
 wire feature_push=active && !rgb && !aborting && (store ? arena_response:read_fire);
 wire [127:0] feature_incoming=store ? arena_data:rdata;
 // Separate the two direction-specific credit cones. Feeding the selected
 // pop back into store_room creates a structural request/ready cycle even
 // though store is constant throughout a transaction.
 // Use a third slot already present in each SRL8. Credits now depend only
 // on registered occupancy, breaking both RAM->AXI and AXI->RAM READY chains.
 // A full queue resumes one edge after pop; the spare slot absorbs that
 // feedback delay without lowering steady-state one-word-per-clock service.
 assign store_room=({1'b0,feature_count}+{2'd0,pending}) < 3'd3;
 assign load_room=feature_count < 2'd3;
 assign feature_valid=feature_count!=0;
 assign busy=state!=IDLE;assign cmd_ready=state==IDLE && !reset && !cancel;
 // Publish a successful transfer on its final acknowledged edge. Error
 // recovery keeps the original DRAIN/DONE sequence and held AXI obligations.
 wire drained=!pending && issued==request_count;
 wire finish_b=state==B && bvalid && bresp==0 && remaining==0 && drained && !aborting;
 wire finish_drain=state==DRAIN && drained && !aborting;
 assign done_valid=!reset && (state==DONE || finish_b || finish_drain);
// Shorter writes bound capture blocking; reads retain128 to avoid extra
 // native commands. Payload/order/4KB and outstanding obligations are unchanged.
 function automatic [8:0] choose(input [31:0] a,input [15:0] beats,input write_direction);
  reg [12:0] boundary;reg [15:0] n;
  begin boundary=(13'd4096-{1'b0,a[11:0]})>>4;n=write_direction && beats>64 ? 16'd64:beats>128 ? 16'd128:beats;choose=n>boundary ? boundary[8:0]:n[8:0];end
 endfunction
 wire [6:0] rgb_unit=second ? 7'd3:7'd2;
 // One outstanding physical RAM token; a returned token can be replaced on
 // this same edge. The buffer, rather than another outstanding address, owns
 // all C32 reorder state.
 wire arena_offer=active && !aborting && issued<request_count && (!pending || arena_response) &&
  (arena_held || (rgb ? (store ? rgb_count<=10: {1'b0,rgb_count}>=rgb_unit):
   (store ? store_room:feature_valid)));
 assign arena_fire=arena_offer && ar[bank];
 wire [127:0] arena_write_data=rgb ? {40'd0,rgb_front[47:24],40'd0,rgb_front[23:0]}:
  feature_word;
 for(genvar b=0;b<6;b=b+1)begin:PORT
  assign av[b]=arena_offer && bank==3'(b);assign aa[b*10+:10]=base+issued[9:0];
  assign ah[b*2+:2]=store ? 2'b00:(!rgb || second ? 2'b11:2'b01);
  assign ad[b*128+:128]=store ? 128'd0:arena_write_data;
  assign sr[b]=active && !aborting && pending && bank==3'(b);
 end
 assign araddr=address;assign arlen=burst_size[7:0]-8'd1;assign arvalid=state==AR && !reset;
 assign awaddr=address;assign awlen=burst_size[7:0]-8'd1;assign awvalid=state==AW && !reset;
 // Consume the old slot and accept its next generation on the same edge.
 // The old RAM/W beat reads pre-edge registers; NBA writes install new data.
 assign rready=state==R && !reset && (aborting || (rgb ? rgb_count<=8:load_room));
 assign read_fire=rvalid && rready;
 wire [6:0] read_bytes=first_read ? 7'd16-{3'd0,prefix}:7'd16;
 // Byte alignment is folded into the RGB bank input index below.
 wire [15:0] byte_position={2'd0,beat_index,4'd0};
 wire [16:0] end_byte={13'd0,prefix}+{2'd0,payload_bytes};
 wire [6:0] beat_bytes=({1'b0,byte_position}+17'd16>end_byte) ? 7'(end_byte-{1'b0,byte_position}):7'd16;
 wire have_write=rgb ? {rgb_count,1'b0}>=beat_bytes:feature_valid;
 wire [127:0] candidate_wdata=rgb ? rgb_front:feature_word;
 reg [15:0] candidate_wstrb;
 always @*begin
  candidate_wstrb=0;
  if(!rgb)begin
   if(!aborting)candidate_wstrb={((({1'b0,beat_index[9:2],1'b0}+10'd1)<({1'b0,columns}+{9'd0,phase})) ? 8'hff:8'h00),((phase && beat_index<4) ? 8'h00:8'hff)};
  end else if(!aborting)begin
   // Whole beats between the two edges need no per-byte wide comparison.
   candidate_wstrb=(beat_index==0 ? (16'hffff<<prefix):16'hffff) &
    ({1'b0,byte_position}+17'd16>end_byte ? (16'hffff>>(5'd16-{1'b0,end_byte[3:0]})):16'hffff);
  end
 end
 assign wdata=(wheld ? !held_zero:!aborting) ? candidate_wdata:128'd0;
 assign wstrb=wheld ? held_wstrb:candidate_wstrb;
 assign wvalid=state==W && !reset && (wheld || aborting || have_write);
 assign wlast=burst_left==1;assign write_fire=wvalid && wready;
 assign bready=state==B && !reset;
 // Transaction handshakes arrive through arbitration. Precompute arithmetic
 // from the held state, and use late handshake bits only as mux selects.
 wire drop_event=store ? (write_fire && !aborting):arena_fire;
 wire [6:0] drop_amount=store ? ((beat_bytes+7'd1)>>1):rgb_unit;
 wire [6:0] drop=drop_event ? drop_amount:7'd0;
 wire [6:0] compact_count={1'b0,rgb_count}-drop;
 wire append_rgb=rgb && !aborting && (store ? arena_response:read_fire);
 wire [6:0] append_size=store ? (pending_second ? 7'd3:7'd2):(read_bytes>>1);
 wire [127:0] append_data=store ? {80'd0,arena_data[87:64],arena_data[23:0]}:rdata;
 wire [6:0] count_unchanged={1'b0,rgb_count};
 wire [6:0] count_removed=count_unchanged-drop_amount;
 wire [6:0] count_added=count_unchanged+append_size;
 wire [6:0] count_replaced=count_removed+append_size;
 wire [6:0] next_count=drop_event ? (append_rgb ? count_replaced:count_removed):
  (append_rgb ? count_added:count_unchanged);
 wire count_bad=drop_event ? (append_rgb ? count_replaced>16:count_removed>16):
  (append_rgb ? count_added>16:count_unchanged>16);
 for(genvar rb=0;rb<8;rb=rb+1)begin:RGB_BANK
  wire [3:0] read_slot=rgb_head+4'((rb-int'(rgb_head[2:0]))&7);
  wire [2:0] read_skip=!store && first_read ? {prefix[3:2],1'b0}:3'd0;
  wire [2:0] input_lane=3'(rb)-rgb_tail[2:0]+read_skip;
  wire [3:0] distance0={1'b0,3'(rb)}-rgb_tail;
  wire [3:0] distance1={1'b1,3'(rb)}-rgb_tail;
  wire write0={3'd0,distance0}<append_size;
  wire write1={3'd0,distance1}<append_size;
  wire put=active && append_rgb && (write0 || write1);
  // Two generations per halfword bank live in one SRL per payload bit.
  // Metadata, not payload, resets. Bytes outside WSTRB are immaterial.
  wire [2:0] tap=read_slot[3]==rgb_latest_slot[rb] ? 3'd7:3'd6;
  for(genvar bitn=0;bitn<16;bitn=bitn+1)begin:BIT
   wire q;
   EFX_SRL8 storage(.D(append_data[input_lane*16+bitn]),.A(tap),.CLK(clk),.CE(put),.Q(q),.Q7());
   assign rgb_bank_window[rb*16+bitn]=~q;
  end
  always @(posedge clk)begin
   if(reset || (cmd_valid && cmd_ready))rgb_latest_slot[rb]<=0;
   else if(put)rgb_latest_slot[rb]<=write1;
  end
 end
 for(genvar lane=0;lane<8;lane=lane+1)begin:RGB_WINDOW
  wire [2:0] rb=rgb_head[2:0]+3'(lane);
  assign rgb_front[lane*16+:16]=rgb_bank_window[rb*16+:16];
 end
 for(genvar bitn=0;bitn<128;bitn=bitn+1)begin:FEATURE_BIT
  wire q;
  EFX_SRL8 payload(.D(feature_incoming[bitn]),.A(3'd0-{1'b0,feature_count}),
   .CLK(clk),.CE(feature_push),.Q(q),.Q7());
  assign feature_word[bitn]=~q;
 end
 always @(posedge clk)begin
  if(reset || (cmd_valid && cmd_ready))feature_count<=0;
  else if(active && !rgb && !aborting)case({feature_push,feature_pop})
   2'b10:feature_count<=feature_count+1'b1;
   2'b01:feature_count<=feature_count-1'b1;
   default:;
  endcase
 end
`ifndef SYNTHESIS
 always @(posedge clk)if(!reset && active && !rgb && !aborting)begin
  if(feature_pop && feature_count==0)$fatal(1,"DMA_FEATURE_QUEUE_UNDERFLOW");
  if(feature_count>3 || (feature_push && !feature_pop && feature_count==3))$fatal(1,"DMA_FEATURE_QUEUE_OVERFLOW");
 end
`endif
 always @(posedge clk)begin
  if(reset)begin
   state<=IDLE;bank<=0;store<=0;rgb<=0;base<=0;columns<=0;address<=0;payload_bytes<=0;
   request_count<=0;issued<=0;remaining<=0;beat_index<=0;burst_left<=0;burst_size<=0;prefix<=0;first_read<=0;
   pending<=0;pending_second<=0;phase<=0;arena_held<=0;rgb_head<=0;rgb_tail<=0;rgb_count<=0;
   error<=0;cycles<=0;arena_words<=0;axi_beats<=0;wheld<=0;held_zero<=0;held_wstrb<=0;
  end else begin
   if(active)begin
    cycles<=cycles+1'b1;
    if(cancel || (arena_response && sw[bank]==store) || (rgb && count_bad))error<=1;
    if(wvalid && !wready)begin
     wheld<=1;held_wstrb<=wstrb;
     // A drain-only zero beat must remain zero on subsequent stalled cycles.
     // Normal held payload still belongs to the frozen RGB/feature queue.
     if(!wheld)held_zero<=aborting;
    end
    if(write_fire)wheld<=0;
    if(rgb && !aborting)begin
     rgb_count<=next_count[5:0];if(drop_event)rgb_head<=rgb_head+drop_amount[3:0];
     if(append_rgb)rgb_tail<=rgb_tail+append_size[3:0];
    end
    if(arena_response)pending<=0;
    if(arena_offer && !ar[bank])arena_held<=1;
    if(arena_fire)begin
     pending<=1;arena_held<=0;pending_second<=second;issued<=issued+11'd1;
     arena_words<=arena_words+(rgb ? (second ? 32'd2:32'd1):
      32'd2-32'(phase && issued<4)-32'(({1'b0,issued[10:2],1'b0}+11'd1)>=({2'd0,columns}+{10'd0,phase})));
    end
    if(aborting)begin pending<=0;arena_held<=0;end
   end
   case(state)
    IDLE:if(cmd_valid && cmd_ready)begin
     integer size,nbeats,nrequests;reg [31:0] aligned_address;
     nrequests=cmd_rgb ? ((int'(cmd_columns)+1)>>1):(((int'(cmd_columns)+int'(cmd_phase)+1)>>1)<<2);
     aligned_address=cmd_rgb ? {cmd_address[31:4],4'd0}:{cmd_address[31:6],6'd0};
     size=cmd_rgb ? int'(cmd_columns)*3:nrequests*16;
     nbeats=cmd_rgb ? (int'(cmd_address[3:0])+size+15)/16:nrequests;
     store<=cmd_store;rgb<=cmd_rgb;bank<=cmd_bank;base<=cmd_base;columns<=cmd_columns;prefix<=cmd_rgb ? {cmd_address[3:2],2'd0}:4'd0;
     phase<=cmd_phase;address<=aligned_address;payload_bytes<=15'(size);request_count<=11'(nrequests);issued<=0;remaining<=11'(nbeats);
     beat_index<=0;burst_size<=choose(aligned_address,16'(nbeats),cmd_store);burst_left<=choose(aligned_address,16'(nbeats),cmd_store);
     first_read<=1;rgb_count<=cmd_store ? {3'd0,cmd_address[3:2],1'b0}:6'd0;rgb_head<=0;
     rgb_tail<=cmd_store ? {1'b0,cmd_address[3:2],1'b0}:4'd0;pending<=0;arena_held<=0;
     error<=0;cycles<=0;arena_words<=0;axi_beats<=0;wheld<=0;
     if(cmd_address[1:0]!=0 || (cmd_rgb ? cmd_phase:(cmd_address[4:0]!=0 || cmd_phase!=cmd_address[5])) ||
        cmd_columns==0 || cmd_bank>=6 || (cmd_rgb ? cmd_groups!=1:cmd_groups!=4) ||
        cmd_address[31:28]!=0 || ({1'b0,(cmd_rgb ? cmd_address:aligned_address)}+33'(size))>33'h010000000 ||
        int'(cmd_base)+nrequests>1024)begin error<=1;state<=DONE;end
     else state<=cmd_store ? AW:AR;
    end
    AR:if(arready)state<=R;
    R:if(read_fire)begin
     first_read<=0;axi_beats<=axi_beats+32'd1;beat_index<=beat_index+10'd1;remaining<=remaining-11'd1;burst_left<=burst_left-9'd1;
     if(rresp!=0 || rlast!=(burst_left==1))error<=1;
     if(burst_left==1)begin
      if(remaining==1 || aborting || rresp!=0)state<=DRAIN;
      else begin address<=address+{19'd0,burst_size,4'd0};burst_size<=choose(address+{19'd0,burst_size,4'd0},remaining-16'd1,1'b0);burst_left<=choose(address+{19'd0,burst_size,4'd0},remaining-16'd1,1'b0);state<=AR;end
     end
    end
    AW:if(awready)state<=W;
    W:if(write_fire)begin
     axi_beats<=axi_beats+32'd1;beat_index<=beat_index+10'd1;remaining<=remaining-11'd1;burst_left<=burst_left-9'd1;
     if(burst_left==1)state<=B;
    end
    B:if(bvalid)begin
     if(bresp!=0)error<=1;
     if(finish_b)state<=done_ready ? IDLE:DONE;
     else if(remaining==0 || aborting || bresp!=0)state<=DRAIN;
     else begin address<=address+{19'd0,burst_size,4'd0};burst_size<=choose(address+{19'd0,burst_size,4'd0},remaining,1'b1);burst_left<=choose(address+{19'd0,burst_size,4'd0},remaining,1'b1);state<=AW;end
    end
    DRAIN:if(!pending && (aborting || issued==request_count))state<=finish_drain && done_ready ? IDLE:DONE;
    DONE:if(done_ready)state<=IDLE;
    default:state<=IDLE;
   endcase
  end
 end
endmodule
