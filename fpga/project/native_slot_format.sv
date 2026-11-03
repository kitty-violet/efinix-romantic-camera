// Frame format is attached to the owner allocation, not the live style key.
// Native storage is separate from the five unchanged 720p input/result pairs.
module native_slot_format(
 input wire clk,reset,
 input wire capture_start,capture_native,input wire[1:0]capture_bank,
 input wire dispatch,input wire[1:0]compute_bank,
 input wire real_frame_valid,input wire[1:0]display_bank,
 output wire cnn_dispatch,frame_valid,display_native,
 output wire[31:0]capture_native_address,display_native_address
);
 reg[2:0]native_slots;
 reg native_job,native_done;
 function automatic[31:0]address(input[1:0]bank);
  case(bank)
   0:address=32'h06c00000;
   1:address=32'h07200000;
   2:address=32'h07800000;
   default:address=0;
  endcase
 endfunction
 wire compute_native=compute_bank<3 && native_slots[compute_bank];
 assign display_native=display_bank<3 && native_slots[display_bank];
 assign capture_native_address=address(capture_bank);
 assign display_native_address=address(display_bank);
 assign cnn_dispatch=dispatch && !compute_native;
 // Existing completion/abort arbitration remains downstream of this signal.
 assign frame_valid=native_job?native_done:real_frame_valid;
 always @(posedge clk)begin
  if(reset)begin native_slots<=0;native_job<=0;native_done<=0;end
  else begin
   if(capture_start && capture_bank<3)native_slots[capture_bank]<=capture_native;
   native_done<=dispatch && compute_native;
   if(dispatch)native_job<=compute_native;
  end
 end
endmodule
