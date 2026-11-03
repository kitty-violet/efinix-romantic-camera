// Write-domain capacity reservation using existing synchronized FIFO pointers.
// The FIFO has a binary wrap bit and Gray-coded lower 12 address bits.
// No new CDC crossing, RAM, reset of outstanding reads or timeout exception.
module native_read_credit(
 input wire clk,reset,
 input wire [12:0] write_gray,read_gray_synced,
 output reg room_short,room_long
);
 reg [12:0] write_bin,read_bin,used_tokens;
 function automatic [12:0] decode(input [12:0] g);
  integer i;
  begin
   decode[12]=g[12];decode[11]=g[11];
   for(i=10;i>=0;i=i-1)decode[i]=decode[i+1]^g[i];
  end
 endfunction
 // Separate decode, distance and comparison to keep the added cone shallow.
 // At an admission edge, the pointer sample is at most 3 writes old.
 // At most 6 unpacked pixels, 2 bypass pixels and 1 frame header remain;
 // 704 >= ceil(128*16/3)+12, 192 >= ceil(32*16/3)+12.
 // Synchronized reads lag true reads, so they only withhold safe capacity.
 always @(posedge clk)begin
  if(reset)begin
   write_bin<=0;read_bin<=0;used_tokens<=13'd4096;
   room_short<=0;room_long<=0;
  end else begin
   write_bin<=decode(write_gray);read_bin<=decode(read_gray_synced);
   used_tokens<=write_bin-read_bin;
   room_short<=used_tokens<=13'd3904;
   room_long<=used_tokens<=13'd3392;
  end
 end
endmodule
