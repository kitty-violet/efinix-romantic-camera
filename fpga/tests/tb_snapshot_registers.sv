`timescale 1ns/1ps
module tb_snapshot_registers;
 reg wc=0,rc=0,source_running=1;
 real wh=6.734,rh=3.333;
 always begin #(wh); if(source_running)wc=~wc; else wc=0; end
 always #(rh)rc=~rc;
 reg request=0,req_meta=0,req_sync=0,ack=0,ack_meta=0,ack_sync=0;
 reg busy=0;
 wire we=req_sync^ack;
 wire re=busy && !(ack_sync^request);
 reg[39:0] incoming=0;
 wire[39:0] reference,candidate;
 reg[39:0] captured=0;
 integer writes=0,reads=0,comparisons=0;
 time last_write=0;
 always @(negedge wc) incoming<={$urandom,$urandom};
 // These equations and the six synchronizer/ack FF links are checked against
 // the actual native8h mapped graph by the Python runner.
 always @(posedge wc)begin
  req_meta<=request;req_sync<=req_meta;
  if(we)begin ack<=req_sync;captured<=incoming;writes<=writes+1;last_write=$time;end
 end
 always @(posedge rc)begin
  ack_meta<=ack;ack_sync<=ack_meta;
  if(re)begin
   if($time-last_write<12)$fatal(1,"insufficient bundled-data settling");
   busy<=0;reads<=reads+1;
   #0.01;
   if(candidate!==captured || reference!==captured)$fatal(1,"coherent capture mismatch");
  end
 end
 for(genvar i=0;i<2;i=i+1)begin:B
  EFX_RAM10 #(.READ_WIDTH(20),.WRITE_WIDTH(20),.OUTPUT_REG(0),.WRITE_MODE("READ_UNKNOWN")) ref_ram(
   .WCLK(wc),.WCLKE(1'b1),.WE({we,we}),.WADDREN(1'b1),.WADDR(9'd0),.WDATA(incoming[i*20+:20]),
   .RCLK(rc),.RE(re),.RST(1'b0),.RADDREN(1'b1),.RADDR(9'd0),.RDATA(reference[i*20+:20]));
  zeroaddr_snapshot20 impl(.wclk(wc),.we(we),.wdata(incoming[i*20+:20]),
   .rclk(rc),.re(re),.rdata(candidate[i*20+:20]));
 end
 always @(negedge rc)begin
  if(reference!==candidate)$fatal(1,"vendor RAM mismatch");
  comparisons=comparisons+1;
 end
 task launch;
  begin @(negedge rc);if(busy)$fatal(1,"test overlap");request=~request;busy=1;end
 endtask
 initial begin
  repeat(6)@(negedge rc);
  if(candidate!==0 || reference!==0)$fatal(1,"initial data mismatch");
  for(integer i=0;i<500;i=i+1)begin
   // Different ratios/phases, data changes continuously; producer clock may
   // stop while the request is in flight. No completion is fabricated.
   wh=(i%3==0)?6.734:((i%3==1)?23.17:2.91);
   if(i%17==0)begin
    @(negedge wc);source_running=0;launch();
    repeat(40)@(negedge rc);
    if(!busy)$fatal(1,"completed with source clock stopped");
    source_running=1;
   end else launch();
   wait(!busy);repeat(i%5+2)@(negedge rc);
  end
  if(writes!=500 || reads!=500)$fatal(1,"transaction count mismatch");
  $display("SNAPSHOT_REG_PASS writes=%0d reads=%0d comparisons=%0d clock_stop_cases=30",writes,reads,comparisons);
  $finish;
 end
 initial begin #2000000;$fatal(1,"timeout");end
endmodule
