// Read-only, mode-0 SPI 03h engine. Never emits WREN/erase/program commands.
// SCLK is ordinary registered data; all logic uses clk, not a fabric clock.
// A low SCLK pause between words provides lossless downstream backpressure.
module spi_flash_read207 #(
 parameter integer HALF_PERIOD=16
)(
 input wire clk,reset,start,input wire[23:0] address,input wire[23:0] byte_count,
 output wire start_ready,output reg busy,done,error,
 output reg cs_n,sclk,mosi,input wire miso,
 output wire[31:0] data,output reg valid,input wire ready
);
 localparam CW=$clog2(HALF_PERIOD);
 reg[CW-1:0] tick;
 // Command TX and data RX are disjoint phases of one SPI transaction.
 // One shift register serves both; a fixed byte permutation preserves LE words.
 reg[31:0] shift_word;
 reg[5:0] command_bits;
 reg[4:0] bit_index;
 reg[21:0] words_remaining;
 assign data={shift_word[7:0],shift_word[15:8],shift_word[23:16],shift_word[31:24]};
 assign start_ready=!busy && !valid;
 always @(posedge clk)begin
  if(reset)begin
   tick<=0;command_bits<=0;bit_index<=0;
   valid<=0;busy<=0;done<=0;error<=0;cs_n<=1;sclk<=0;mosi<=0;
  end else begin
   done<=0;
   if(start && start_ready)begin
    if(byte_count==0 || byte_count[1:0]!=0 || {1'b0,address}+{1'b0,byte_count}>25'h0800000)begin error<=1;done<=1;end
    else begin
     busy<=1;error<=0;cs_n<=0;sclk<=0;mosi<=0;tick<=0;
     shift_word<={8'h03,address};command_bits<=32;words_remaining<=byte_count[23:2];
     bit_index<=0;
    end
   end else if(busy)begin
    if(valid)begin
     if(ready)begin
      valid<=0;
      if(words_remaining==0)begin busy<=0;cs_n<=1;done<=1;end
     end
    end else if(tick==HALF_PERIOD-1)begin
     tick<=0;sclk<=!sclk;
     if(sclk)begin
      // End of high half: output has settled for a full half-period.
      if(command_bits!=0)begin
       shift_word<={shift_word[30:0],1'b0};command_bits<=command_bits-1;mosi<=shift_word[30];
      end else begin
       shift_word<={shift_word[30:0],miso};bit_index<=bit_index+1'b1;
       if(bit_index==31)begin words_remaining<=words_remaining-1'b1;valid<=1;end
      end
     end
    end else tick<=tick+1;
   end
  end
 end
 initial if(HALF_PERIOD<2)$fatal(1,"SPI half-period must allow IO settling");
endmodule
