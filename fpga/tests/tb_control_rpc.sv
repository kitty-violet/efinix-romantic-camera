`timescale 1ns/1ps
module tb_control_rpc;
 reg cpu_restart=0;
 reg reset=1,cpu_clk=0,service_clk=0,service_enable=1;
 real service_half=3.333;
 always #5 cpu_clk=~cpu_clk;
 always begin #(service_half);if(service_enable)service_clk=~service_clk;end
 reg psel=0,penable=0,pwrite=0;
 reg[15:0]paddr=0;reg[31:0]pwdata=0;
 wire pready,pslverror,protocol_error;wire[31:0]prdata;
 wire request_valid,request_write,response_ready;
 wire[15:0]request_address;wire[31:0]request_data;
 reg request_ready=0,pending=0;reg[7:0]delay_left=0;
 reg[31:0]held_response=0;reg held_error=0;
 wire immediate=request_address[3];
 wire accepted=request_valid && request_ready;
 wire response_valid=(accepted && immediate) || (pending && delay_left==0);
 wire[31:0]response_data=pending?held_response:(request_data^{16'hbeef,request_address}^{31'd0,request_write});
 wire response_error=pending?held_error:request_address[10];
 integer accepted_count=0,completed_count=0,tests=0;
 control_rpc_bridge #(.RPC_TIMEOUT(128)) dut(.*);
 always @(posedge service_clk or posedge reset)begin
  if(reset)begin pending<=0;delay_left<=0;request_ready<=0;end
  else begin
   request_ready<=!pending && $urandom_range(0,3)!=0;
   if(accepted)begin
    accepted_count<=accepted_count+1;
    if(!immediate)begin
     pending<=1;held_response<=request_data^{16'hbeef,request_address}^{31'd0,request_write};
     held_error<=request_address[10];delay_left<=$urandom_range(0,17);
    end
   end
   if(pending && delay_left!=0)delay_left<=delay_left-1'b1;
   if(response_valid && response_ready)begin pending<=0;completed_count<=completed_count+1;end
  end
 end
 // The destination must retain an entire atomic command under backpressure.
 reg stalled=0;reg[48:0]stalled_command;
 always @(posedge service_clk)begin
  if(reset)stalled<=0;
  else begin
   if(stalled && (!request_valid || {request_write,request_address,request_data}!==stalled_command))
    $fatal(1,"command changed while stalled");
   stalled<=request_valid && !request_ready;
   stalled_command<={request_write,request_address,request_data};
  end
 end
 task automatic access(input bit wr,input[15:0]addr,input[31:0]data,input bit timer_access=0);
  reg[31:0]expected;integer age,old_count;
  begin
   old_count=accepted_count;
   @(negedge cpu_clk);psel=1;penable=0;pwrite=wr;paddr=addr;pwdata=data;
   @(negedge cpu_clk);penable=1;age=0;
   do begin @(posedge cpu_clk);age=age+1;if(age>500)$fatal(1,"APB timeout");end while(!pready);
   if(timer_access)begin
    if(pslverror!==wr)$fatal(1,"timer write protection");
    if(accepted_count!=old_count)$fatal(1,"timer crossed CDC");
   end else begin
    expected=data^{16'hbeef,addr}^{31'd0,wr};
    if(prdata!==expected || pslverror!==addr[10])$fatal(1,"response mismatch %h %h %h",addr,prdata,expected);
    if(accepted_count!=old_count+1)$fatal(1,"lost/duplicate request");
   end
   tests=tests+1;
   @(negedge cpu_clk);penable=0;psel=0;
   repeat($urandom_range(0,3))@(negedge cpu_clk);
  end
 endtask
 task automatic restart;
  begin
   reset=1;psel=0;penable=0;
   #27;reset=0;
   repeat(20)@(posedge cpu_clk);
   if(request_valid || response_valid || protocol_error)$fatal(1,"stale transfer after reset");
  end
 endtask
 initial begin
  #12;reset=0;
  // Destination clock absent at power-up: no source request is accepted early.
  service_enable=0;
  fork
   begin access(1,16'h0040,32'h12345678);end
   begin #300;if(pready || request_valid)$fatal(1,"startup barrier failed");service_enable=1;end
  join
  for(integer phase=0;phase<3;phase=phase+1)begin
   service_half=phase==0?3.333:(phase==1?7.7:2.1);
   for(integer n=0;n<240;n=n+1)begin
    access($urandom_range(0,1),16'h1000+n*4,$urandom);
    if(n%31==0)begin access(0,16'h78,0,1);access(1,16'h78,0,1);end
   end
   restart();
  end
  // Reset while a real APB request is outstanding; both endpoints are aborted.
  @(negedge cpu_clk);psel=1;pwrite=0;paddr=16'h1000;
  @(negedge cpu_clk);penable=1;
  wait(accepted);#1;restart();
  access(0,16'h40,32'h11223344);
  if(protocol_error)$fatal(1,"unexpected protocol fault");
  // CPU-local reset while service clock is stopped must not reset mailboxes.
  @(negedge cpu_clk);psel=1;penable=0;pwrite=1;paddr=16'h40;pwdata=32'h11112222;
  @(negedge cpu_clk);penable=1;
  wait(pending);@(negedge service_clk);service_enable=0;
  @(negedge cpu_clk);cpu_restart=1;psel=0;penable=0;
  repeat(12)@(posedge cpu_clk);
  if(pready || !protocol_error)$fatal(1,"local reset containment");
  @(negedge cpu_clk);cpu_restart=0;service_enable=1;
  access(0,16'h44,32'h33334444);
  if(!protocol_error)$fatal(1,"restart fault not sticky");
  restart();
  // Late response after a CPU-side timeout is drained, not returned to next APB.
  @(negedge cpu_clk);psel=1;penable=0;pwrite=0;paddr=16'h40;pwdata=32'h55556666;
  @(negedge cpu_clk);penable=1;
  wait(pending);@(negedge service_clk);service_enable=0;
  do @(posedge cpu_clk);while(!pready);
  if(!pslverror)$fatal(1,"RPC timeout did not complete with error");
  @(negedge cpu_clk);psel=0;penable=0;
  repeat(15)@(posedge cpu_clk);
  if(!protocol_error)$fatal(1,"RPC timeout not sticky");
  service_enable=1;access(0,16'h44,32'h77778888);
  restart();access(0,16'h40,32'h12345678);
  $display("PASS RPC_CDC_ATOMIC transactions=%0d accepted=%0d responses=%0d",tests,accepted_count,completed_count);
  $finish;
 end
 initial begin #1000000;$fatal(1,"watchdog");end
endmodule
