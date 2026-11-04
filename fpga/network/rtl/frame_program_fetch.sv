// Candidate: one shared CRC32 unit starts on R handshake; last CRC edge
// publishes the record. No extra wide CRC hardware or unbounded lookahead.
// S4R1 descriptor transport. Two compute and two DMA record slots, one AXI reader,
// one time-shared CRC32/32-bit unit. No unbounded program memory or TB issuer.
// Read-only program lives in the temporary DDR region. Frame publication MUST
// wait for verified, both command engines complete, and all output B responses.
// Cancel/error suppress consumers immediately, but never withdraw an exposed AR;
// an accepted read is drained to actual RLAST. Only common reset revokes AXI.
module frame_program_fetch #(
 parameter [31:0] COMPUTE_BASE=32'h04000000,DMA_BASE=32'h04100000,
 parameter integer RESIDENT_PROGRAMS=0,
 parameter integer MAX_COMPUTE=6546,MAX_DMA=8277
)(
 input wire clk,reset,cancel,graceful_abort,
 input wire [2:0] active_style_slot,
 input wire start_valid,output wire start_ready,
 input wire [13:0] compute_records,dma_records,
 input wire [31:0] compute_crc32,dma_crc32,
 output wire cvalid,input wire cready,output wire [511:0] cdata,
 output wire next_valid,output wire [511:0] next_data,
 output wire dvalid,input wire dready,output wire [127:0] ddata,
 output wire [31:0] araddr,output wire [7:0] arlen,
 output wire arvalid,input wire arready,
 input wire [127:0] rdata,input wire [1:0] rresp,
 input wire rlast,rvalid,output wire rready,
 output wire busy,quiesced,output reg verified,error,
 output reg [3:0] fault_code,
 output wire [13:0] compute_fetched,dma_fetched
);
 localparam [2:0] IDLE=0,ADDRESS=1,READ=2,CRC=3,COMMIT=4,DRAIN=5;
 reg [2:0] state;
 reg active,aborting,owner,prefer_dma;
 reg [13:0] ctotal,dtotal,cfetched,dfetched;
 reg [31:0] ccrc,dcrc,cexpected,dexpected;
 reg [1:0] ccount,dcount,beat,crc_step;
 reg [511:0] latest_compute;
 reg [3:0] partial_compute;reg partial_dma;
 reg [127:0] crc_word;
 reg [2:0] style_slot_q;
 wire [31:0] compute_base_live=RESIDENT_PROGRAMS ? (32'h04800000+{9'd0,style_slot_q,20'd0}):COMPUTE_BASE;
 reg [31:0] address_q;reg [7:0] length_q;
 wire stopped=aborting || cancel || error;
 wire cpop=cvalid && cready,dpop=dvalid && dready;
 wire record_last=state==CRC && crc_step==3 && (owner || beat==3);
 wire cpush=record_last && !owner && !stopped;
 wire dpush=record_last && owner && !stopped;
 wire cwanted=cfetched<ctotal && ccount<2;
 wire dwanted=dfetched<dtotal && dcount<2;
 wire choose_dma=dwanted && (!cwanted || prefer_dma);
 assign next_data=latest_compute;
 assign cvalid=active && ccount!=0 && !stopped && !reset;
 assign next_valid=active && ccount==2 && !stopped && !reset;
 assign dvalid=active && dcount!=0 && !stopped && !reset;
 assign araddr=address_q;assign arlen=length_q;
 assign arvalid=state==ADDRESS && !reset;
 assign rready=(state==READ || state==DRAIN) && !reset;
 assign busy=active || state!=IDLE;
 assign quiesced=state==IDLE;
 assign start_ready=!busy && !reset && !cancel;
 assign compute_fetched=cfetched;assign dma_fetched=dfetched;
 // Reflected IEEE CRC32, identical to zlib.crc32 over little-endian bytes.
 function automatic [31:0] crc32word(input [31:0] initial_crc,input [31:0] word);
  reg [31:0] v;begin
   v=initial_crc;
   for(integer i=0;i<32;i=i+1)v=(v>>1)^((v[0]^word[i]) ? 32'hedb88320:32'd0);
   crc32word=v;
  end
 endfunction
 wire [31:0] updated_crc=crc32word(owner ? dcrc:ccrc,state==READ ? rdata[31:0]:crc_word[31:0]);
 wire limits_ok=(!RESIDENT_PROGRAMS || active_style_slot<6) && compute_records!=0 && {18'd0,compute_records}<=MAX_COMPUTE &&
  dma_records!=0 && {18'd0,dma_records}<=MAX_DMA &&
  COMPUTE_BASE[5:0]==0 && DMA_BASE[3:0]==0 &&
  COMPUTE_BASE>=32'h04000000 && COMPUTE_BASE+MAX_COMPUTE*64<=32'h04100000 &&
  DMA_BASE>=32'h04100000 && DMA_BASE+MAX_DMA*16<=32'h04200000;
 always @(posedge clk)begin
  if(reset)begin
   state<=IDLE;active<=0;aborting<=0;owner<=0;prefer_dma<=0;
   ctotal<=0;dtotal<=0;cfetched<=0;dfetched<=0;
   ccrc<=32'hffffffff;dcrc<=32'hffffffff;cexpected<=0;dexpected<=0;
   ccount<=0;dcount<=0;
   beat<=0;crc_step<=0;crc_word<=0;address_q<=0;length_q<=0;style_slot_q<=0;
   verified<=0;error<=0;fault_code<=0;
  end else begin
   if(cancel && active)begin aborting<=1;verified<=0;if(!error)fault_code<=4'd1;end
   // Match native221: extend only the existing aborting FF enable.
   // It clears on start; it must not hold the external cancel/start_ready gate.
   if(graceful_abort && active)aborting<=1;
   case({cpush,cpop})2'b10:ccount<=ccount+1'b1;2'b01:ccount<=ccount-1'b1;default:;endcase
   case({dpush,dpop})2'b10:dcount<=dcount+1'b1;2'b01:dcount<=dcount-1'b1;default:;endcase
   if(cpush)cfetched<=cfetched+1'b1;
   if(dpush)dfetched<=dfetched+1'b1;
   if(active && stopped && state==IDLE)begin active<=0;ccount<=0;dcount<=0;verified<=0;end
   if(active && !stopped && cfetched==ctotal && dfetched==dtotal && state==IDLE)begin
    if((~ccrc)!=cexpected || (~dcrc)!=dexpected)begin error<=1;fault_code<=4'd4;verified<=0;end
    else begin verified<=1;if(ccount==0 && dcount==0)active<=0;end
   end
   case(state)
    IDLE:begin
     if(start_valid && start_ready)begin
    style_slot_q<=active_style_slot;
      active<=limits_ok;error<=!limits_ok;fault_code<=limits_ok ? 4'd0:4'd2;
      aborting<=0;verified<=0;ctotal<=compute_records;dtotal<=dma_records;
      cexpected<=compute_crc32;dexpected<=dma_crc32;
      ccrc<=32'hffffffff;dcrc<=32'hffffffff;cfetched<=0;dfetched<=0;
      ccount<=0;dcount<=0;prefer_dma<=0;
     end else if(active && !stopped && (cwanted || dwanted))begin
      owner<=choose_dma;prefer_dma<=!choose_dma;beat<=0;
      address_q<=choose_dma ? DMA_BASE+{14'd0,dfetched,4'd0}:compute_base_live+{12'd0,cfetched,6'd0};
      length_q<=choose_dma ? 8'd0:8'd3;state<=ADDRESS;
     end
    end
    ADDRESS:if(arready)state<=READ;
    READ:if(rvalid)begin
     if(stopped || rresp!=0 || rlast!=(owner || beat==3))begin
      if(rresp!=0 || rlast!=(owner || beat==3))begin
       error<=1;if(!error)fault_code<=rresp!=0 ? 4'd3:4'd5;
      end
      if(!rlast)beat<=beat+1'b1;
      state<=rlast ? IDLE:DRAIN;
     end else begin
      // Consume the first CRC word on the accepted AXI beat itself.
      if(owner)dcrc<=updated_crc;else ccrc<=updated_crc;
      crc_word<={32'd0,rdata[127:32]};crc_step<=1;state<=CRC;
     end
    end
    CRC:begin
     if(owner)dcrc<=updated_crc;else ccrc<=updated_crc;
     crc_word<={32'd0,crc_word[127:32]};crc_step<=crc_step+1'b1;
     if(crc_step==3)begin
      if(owner || beat==3)state<=IDLE;
      else begin beat<=beat+1'b1;state<=stopped ? DRAIN:READ;end
     end
    end
    COMMIT:state<=IDLE;
    DRAIN:if(rvalid)begin
     if(rresp!=0 || rlast!=(owner || beat==3))begin error<=1;if(!error)fault_code<=rresp!=0 ? 4'd3:4'd5;end
     if(rlast)state<=IDLE;else beat<=beat+1'b1;
    end
    default:begin state<=IDLE;error<=1;fault_code<=4'd6;end
   endcase
  end
 end

 // Each received 128-bit fragment shifts into its own bounded two-record
 // history. Until CRC publishes the record, partial_* extends the physical
 // depth so an existing consumer still sees the old complete front record.
 wire receive_good=state==READ && rvalid && !stopped && rresp==0 && rlast==(owner || beat==3);
 always @(posedge clk)begin
  if(reset || (start_valid && start_ready))begin partial_compute<=0;partial_dma<=0;end
  else begin
   if(cpush)partial_compute<=0;
   if(dpush)partial_dma<=0;
   if(receive_good)begin
    if(owner)partial_dma<=1;
    else begin partial_compute[beat]<=1;latest_compute[beat*128+:128]<=rdata;end
   end
  end
 end
 for(genvar word=0;word<4;word=word+1)begin:COMPUTE_WORD
  wire [2:0] depth={1'b0,ccount}+{2'd0,partial_compute[word]};
  wire [2:0] address=3'd0-depth;
  for(genvar bitn=0;bitn<128;bitn=bitn+1)begin:BIT
   wire q;
   EFX_SRL8 record(.D(rdata[bitn]),.A(address),.CLK(clk),
    .CE(receive_good && !owner && beat==2'(word)),.Q(q),.Q7());
   assign cdata[word*128+bitn]=~q;
  end
 end
 wire [2:0] ddepth={1'b0,dcount}+{2'd0,partial_dma};
 for(genvar bitn=0;bitn<128;bitn=bitn+1)begin:DMA_BIT
  wire q;
  EFX_SRL8 record(.D(rdata[bitn]),.A(3'd0-ddepth),.CLK(clk),
   .CE(receive_good && owner),.Q(q),.Q7());
  assign ddata[bitn]=~q;
 end
 // synthesis translate_off
 always @(posedge clk)if(!reset && active && !stopped)begin
  if(ddepth>2)$fatal(1,"DMA_RECORD_STORAGE_OVERFLOW");
  for(integer word=0;word<4;word=word+1)
   if({1'b0,ccount}+{2'd0,partial_compute[word]}>2)$fatal(1,"COMPUTE_RECORD_STORAGE_OVERFLOW");
 end
 // synthesis translate_on

endmodule
