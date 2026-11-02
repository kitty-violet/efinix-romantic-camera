// Inputs are already synchronized in the released 222 pixel domain.
// Keep the old publication protocol and latch the label only at frame start.
module legacy222_names(
 input wire clk,reset_n,de,frame_start,ready_sync,ready_qualified,
 input wire[2:0] slot_sync,
 input wire[11:0] x,input wire[10:0] y,
 input wire[23:0] source_rgb,output wire[23:0] output_rgb
);
 reg[2:0] frame_slot=0;
 reg frame_ready=0;
 always @(posedge clk or negedge reset_n) begin
  if(!reset_n)begin frame_slot<=0;frame_ready<=0;end
  else if(frame_start)begin
   frame_slot<=slot_sync;
   frame_ready<=ready_sync && ready_qualified;
  end
 end
 // Erase old A1/V14 only; preserve the adjacent existing AI FPS rectangle.
 wire old_label=de && x>=8 && x<32 && y>=32 && y<48;
 wire[23:0] clean_rgb=old_label ? 24'h000000 : source_rgb;
 five_style_name_osd font(.clk(clk),.de(de),.prepared(frame_ready),
  .selected_slot(frame_slot),.x(x),.y(y),.source_rgb(clean_rgb),.output_rgb(output_rgb));
endmodule
