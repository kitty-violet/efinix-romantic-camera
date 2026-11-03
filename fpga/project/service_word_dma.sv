// Bounded control-plane bulk engine. Runs in the existing service/bus domain.
// CPU submits one whole transfer; it never copies individual weight words.
// cmd_* is a single outstanding, response-bearing native register/DDR port.
module service_word_dma #(
 parameter integer BUS_TIMEOUT=4096, DIAG_POLL_LIMIT=4096
)(
 input wire clk,reset,start,input wire[2:0]operation,
 input wire[31:0]source,target,byte_count,
 output wire start_ready,output reg busy,done,error,poisoned,
 output reg[7:0]fault,output wire[31:0]crc_result,word_result,
 output wire cmd_valid,cmd_write,output reg[31:0]cmd_address,cmd_data,
 input wire cmd_ready,rsp_valid,rsp_error,input wire[31:0]rsp_data,
 output wire spi_start,input wire spi_start_ready,
 output wire[23:0]spi_address,spi_length,
 input wire spi_valid,spi_error,input wire[31:0]spi_data,output wire spi_ready
);
 localparam IDLE=0,FLASH_START=1,FETCH=2,READ0=3,CRC0=4,WRITE0=5,
 READ1=6,CRC1=7,INDEX=8,LOW=9,HIGH=10,PUSH=11,STORE_STATUS=12,
 DIAG_INDEX=13,DIAG_GO=14,DIAG_POLL=15,DIAG_META=16,DIAG_TAG=17,
 DIAG_LOW=18,DIAG_HIGH=19,FAILED=20;
 reg[4:0]state;
 reg[2:0]mode;
 reg[31:0]src,dst,lo,crc,shift_word;wire[31:0]hi;
 // Validated transfers are <=1MiB, including the inclusive maximum.
 reg[20:0]left;
 reg[11:0]index;
 reg[2:0]bank,nibble;
 reg[8:0]row;
 reg verifying,waiting,timeout_latched;
 reg[$clog2(BUS_TIMEOUT+1)-1:0]bus_age;
 reg[$clog2(DIAG_POLL_LIMIT+1)-1:0]polls;
 wire bus_phase=state==READ0 || state==READ1 || state==WRITE0 ||
  (state>=INDEX && state<=DIAG_HIGH);
 wire offer=bus_phase && !waiting;
 assign cmd_valid=offer && !reset;
 assign cmd_write=!(state==READ0 || state==READ1 || state==STORE_STATUS ||
  state==DIAG_POLL || state==DIAG_META || state==DIAG_TAG || state==DIAG_LOW || state==DIAG_HIGH);
 wire accepted=cmd_valid && cmd_ready;
 wire reply=rsp_valid && (waiting || accepted);
 assign start_ready=state==IDLE && !error;
 assign crc_result=~crc;
 assign word_result=lo;
 assign spi_start=state==FLASH_START && !reset;
 assign spi_address=src[23:0];assign spi_length={3'd0,left};
 assign spi_ready=state==FETCH && !error && !reset;
 control_dsp_register #(.WIDTH(32)) hi_storage(.clk(clk),.reset(reset),
  .enable(state==READ1 && reply && !(timeout_latched || rsp_error) && !error),.data(rsp_data),.q(hi));
 wire legal_common=byte_count!=0 && byte_count[1:0]==0 && source[1:0]==0 &&
  byte_count<=32'h100000;
 // Upper count bits are rejected by legal_common before these predicates
 // can authorize bulk commands; retain one carry bit on 24-bit addresses.
 wire source_ddr=source[31:24]==8'he1 && {1'b0,source[23:0]}+{4'd0,byte_count[20:0]}<=25'h1000000;
 wire target_ddr=target[31:24]==8'he1 && target[1:0]==0 && {1'b0,target[23:0]}+{4'd0,byte_count[20:0]}<=25'h1000000;
 wire legal_flash=source[31:24]==0 && source[23:0]>=24'h340000 &&
  {1'b0,source[23:0]}+{4'd0,byte_count[20:0]}<=25'h800000;
 wire legal_coefficient_source=source==32'he1200000 ||
  (source>=32'he18c0000 && source<=32'he1dc0000 && source[19:0]==20'hc0000);
 wire legal_native_source=source[1:0]==0 && (source[31:24]==8'he1 ||
  (source>=32'hf8100000 && source<32'hf8100080) ||
  (source>=32'hf9000f00 && source<32'hf9000f40));
 wire legal=(legal_common && ((operation==1 && legal_flash && target_ddr) ||
  (operation==2 && source_ddr && target_ddr) || (operation==3 && source_ddr) ||
  (operation==4 && legal_coefficient_source && byte_count==20480 && target!=0 && target<=65535))) ||
  ((operation==5 || operation==6) && byte_count==4 && legal_native_source);
 function automatic[31:0]crc_nibble(input[31:0]c,input[3:0]d);
  reg[31:0]v;
  begin v=c^{28'd0,d};for(integer k=0;k<4;k=k+1)v=(v>>1)^(v[0]?32'hedb88320:0);crc_nibble=v;end
 endfunction
 wire[31:0]crc_next=crc_nibble(crc,shift_word[3:0]);
 always @*begin
  cmd_address=32'hf8100000;cmd_data=0;
  case(state)
   READ0:cmd_address=src;
   READ1:cmd_address=src+4;
   WRITE0:begin cmd_address=dst;cmd_data=lo;end
   INDEX:begin cmd_address=32'hf8100024;cmd_data={20'd0,index};end
   LOW:begin cmd_address=32'hf8100028;cmd_data=lo;end
   HIGH:begin cmd_address=32'hf810002c;cmd_data=hi;end
   PUSH:begin cmd_address=32'hf8100030;cmd_data=index==2559?3:1;end
   STORE_STATUS:cmd_address=32'hf8100004;
   DIAG_INDEX:begin cmd_address=32'hf8100034;cmd_data={19'd0,bank,1'b0,row};end
   DIAG_GO:begin cmd_address=32'hf8100038;cmd_data=1;end
   DIAG_POLL:cmd_address=32'hf810003c;
   DIAG_META:cmd_address=32'hf810004c;
   DIAG_TAG:cmd_address=32'hf8100068;
   DIAG_LOW:cmd_address=32'hf8100040;
   DIAG_HIGH:cmd_address=32'hf8100044;
   default:begin end
  endcase
 end
 task fail(input[7:0]code,input bit ambiguous);
  begin error<=1;fault<=code;poisoned<=ambiguous;busy<=0;done<=1;state<=FAILED;end
 endtask
 task next_word;
  begin
   src<=src+4;dst<=dst+4;left<=left-4;
   if(left==4)begin busy<=0;done<=1;state<=IDLE;end
   else state<=mode==1?FETCH:READ0;
  end
 endtask
 task next_pair;
  begin
   src<=src+8;index<=index+1'b1;
   if(bank==4)begin bank<=0;row<=row+1'b1;end else bank<=bank+1'b1;
   if(index==2559)begin
    if(verifying)begin busy<=0;done<=1;state<=IDLE;end
    // Every legal coefficient source is 64KiB-aligned. The full20480-byte
    // scan stays inside that64KiB page; reconstruct its base without storing
    // a redundant32-bit shadow address. General bulk modes never use this.
    else begin verifying<=1;src<={src[31:16],16'd0};index<=0;bank<=0;row<=0;state<=READ0;end
   end else state<=READ0;
  end
 endtask
 always @(posedge clk)begin
  if(reset)begin
   state<=IDLE;mode<=0;src<=0;dst<=0;left<=0;lo<=0;
   crc<=32'hffffffff;shift_word<=0;index<=0;bank<=0;row<=0;nibble<=0;
   verifying<=0;waiting<=0;bus_age<=0;polls<=0;timeout_latched<=0;
   busy<=0;done<=0;error<=0;poisoned<=0;fault<=0;
  end else begin
   if(accepted)waiting<=1;
   if(reply)waiting<=0;
   if(bus_phase && !reply)begin
    if(bus_age<BUS_TIMEOUT)bus_age<=bus_age+1'b1;
    if(bus_age==BUS_TIMEOUT-1)begin error<=1;poisoned<=1;fault<=8'hd2;timeout_latched<=1;end
   end else bus_age<=0;
   // A timeout never withdraws an offered native command. Drain its eventual
   // response, retain the failure, and forbid all subsequent transactions.
   if(reply && (timeout_latched || rsp_error))fail(timeout_latched?8'hd2:8'hd1,timeout_latched);
   else if(!error)begin
    case(state)
     IDLE:if(start)begin
      done<=0;fault<=0;
      if(!legal)fail(8'h01,0);
      else begin mode<=operation;src<=source;dst<=operation==6?source:target;left<=byte_count[20:0];
       if(operation==6)lo<=target;
       crc<=32'hffffffff;index<=0;bank<=0;row<=0;verifying<=0;busy<=1;
       state<=operation==1?FLASH_START:operation==6?WRITE0:READ0;end
     end
     FLASH_START:if(spi_start_ready)state<=FETCH;
     FETCH:if(spi_error)fail(8'h02,0);
      else if(spi_valid)begin lo<=spi_data;shift_word<=spi_data;nibble<=0;state<=CRC0;end
     READ0:if(reply)begin
      lo<=rsp_data;
      if(mode==5)begin busy<=0;done<=1;state<=IDLE;end
      else if(mode==4 && verifying)state<=READ1;
      else begin shift_word<=rsp_data;nibble<=0;state<=CRC0;end
     end
     READ1:if(reply)begin
      if(verifying)state<=DIAG_INDEX;
      else begin shift_word<=rsp_data;nibble<=0;state<=CRC1;end
     end
     CRC0,CRC1:begin
      crc<=crc_next;shift_word<=shift_word>>4;nibble<=nibble+1'b1;
      if(nibble==7)begin
       if(mode==4)state<=state==CRC0?READ1:INDEX;
       else if(mode==3)next_word();else state<=WRITE0;
      end
     end
     WRITE0:if(reply)begin
      if(mode==6)begin lo<=rsp_data;busy<=0;done<=1;state<=IDLE;end
      else next_word();
     end
     INDEX:if(reply)state<=LOW;
     LOW:if(reply)state<=HIGH;
     HIGH:if(reply)state<=PUSH;
     PUSH:if(reply)state<=STORE_STATUS;
     STORE_STATUS:if(reply)begin if(rsp_data[3])fail(8'h03,0);else next_pair();end
     DIAG_INDEX:if(reply)state<=DIAG_GO;
     DIAG_GO:if(reply)begin polls<=0;state<=DIAG_POLL;end
     DIAG_POLL:if(reply)begin
      if(rsp_data[0])state<=DIAG_META;
      else if(polls==DIAG_POLL_LIMIT-1)fail(8'h04,0);else polls<=polls+1'b1;
     end
     DIAG_META:if(reply)begin
      if(rsp_data!={19'd0,bank,1'b0,row})fail(8'h05,0);else state<=DIAG_TAG;
     end
     DIAG_TAG:if(reply)begin if(rsp_data!=dst)fail(8'h06,0);else state<=DIAG_LOW;end
     DIAG_LOW:if(reply)begin if(rsp_data!=lo)fail(8'h07,0);else state<=DIAG_HIGH;end
     DIAG_HIGH:if(reply)begin if(rsp_data!=hi)fail(8'h08,0);else next_pair();end
     default:fail(8'hff,0);
    endcase
   end
  end
 end
endmodule
