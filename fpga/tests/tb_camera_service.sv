`timescale 1ns/1ps
module tb_camera_service;
 reg reset=1,service_clk=0,camera_clk=0;
 reg camera_power_ready=1;
 always #3.333 service_clk=~service_clk;always #100 camera_clk=~camera_clk;
 reg camera_request_valid=0,camera_response_ready=0;
 reg[32:0]camera_request=0;
 wire camera_request_ready,camera_response_valid;
 wire[3:0]camera_response_error;wire[2:0]camera_response_error_byte;wire[7:0]camera_response_data;
 tri1 scl,sda;wire scl_o,scl_oen,sda_o,sda_oen;
 reg slave_low=0,hold_scl=0,nack_data=0;
 assign scl=scl_oen?1'bz:scl_o;assign scl=hold_scl?1'b0:1'bz;
 assign sda=sda_oen?1'bz:sda_o;assign sda=slave_low?1'b0:1'bz;
 camera_service_bridge #(.PRESCALE(8),.TIMEOUT_CYCLES(5000)) dut(.scl_i(scl),.sda_i(sda),.*);
 reg[7:0]mem[0:65535];
 integer bit_count=0,byte_count=0,starts=0,stops=0,writes=0,reads=0,tests=0;
 reg active=0,address_match=0,reading=0,ack_byte=0;
 reg[7:0]shift=0,send_byte=0,device=8'h60;reg[15:0]address=0;
 always @(negedge sda)if(scl===1 && !reset)begin
  active=1;bit_count=0;byte_count=0;address_match=0;reading=0;ack_byte=0;starts=starts+1;
 end
 always @(posedge sda)if(scl===1 && !reset && active)begin active=0;stops=stops+1;end
 always @(posedge scl)if(active && !reset)begin
  if(bit_count<8)begin
   shift={shift[6:0],sda};bit_count=bit_count+1;
   if(bit_count==8 && !reading)begin
    if(byte_count==0)begin address_match=(shift&8'hfe)==device;ack_byte=address_match;end
    else ack_byte=address_match;
    if(byte_count==1)address[15:8]=shift;
    if(byte_count==2)address[7:0]=shift;
    if(byte_count==3 && address_match)begin
     ack_byte=!nack_data;if(ack_byte)begin mem[address]=shift;writes=writes+1;end
    end
   end
  end else begin
   if(reading)begin
    if(sda!==1)$fatal(1,"master must NACK final byte");reads=reads+1;reading=0;address_match=0;
   end else if(byte_count==0 && shift[0] && address_match)begin reading=1;send_byte=mem[address];end
   bit_count=0;byte_count=byte_count+1;
  end
 end
 always @(negedge scl)begin
  if(!active || reset)slave_low=0;
  else if(reading)slave_low=bit_count<8?!send_byte[7-bit_count]:0;
  else slave_low=bit_count==8 && ack_byte;
 end
 task automatic transact(input bit rd,input[7:0]dev,input[15:0]addr,input[7:0]data,
                         input[3:0]err,input[2:0]err_byte,input[7:0]expected);
  reg[14:0]held;integer age,old_starts;
  begin
   @(negedge service_clk);camera_request_valid=1;camera_request={rd,addr,dev,data};
   do @(posedge service_clk);while(!camera_request_ready);
   @(negedge service_clk);camera_request_valid=0;
   age=0;
   do begin @(posedge service_clk);age=age+1;if(age>400000)$fatal(1,"camera RPC timeout");end while(!camera_response_valid);
   held={camera_response_error,camera_response_error_byte,camera_response_data};old_starts=starts;
   repeat(25)begin
    @(posedge service_clk);
    if(!camera_response_valid || held!=={camera_response_error,camera_response_error_byte,camera_response_data})$fatal(1,"camera response changed under stall");
   end
   if(camera_response_error!==err || (err!=0 && camera_response_error_byte!==err_byte) ||
      (err==0 && rd && camera_response_data!==expected))$fatal(1,"camera result %h",held);
   if(starts!=old_starts)$fatal(1,"duplicate I2C command");
   @(negedge service_clk);camera_response_ready=1;
   @(negedge service_clk);camera_response_ready=0;tests=tests+1;
   repeat(15)@(posedge camera_clk);
  end
 endtask
 initial begin
  for(integer k=0;k<65536;k=k+1)mem[k]=0;mem['h3107]=8'hcd;mem['h3108]=8'h6b;
  #37;reset=0;
  transact(1,'h60,'h3107,0,0,0,'hcd);
  transact(0,'h60,'h3e08,'h83,0,0,0);
  transact(1,'h60,'h3e08,0,0,0,'h83);
  transact(1,'h64,'h3107,0,1,0,0);
  nack_data=1;transact(0,'h60,'h3e08,'hff,1,3,0);nack_data=0;
  if(mem['h3e08]!='h83)$fatal(1,"NACK write changed memory");
  hold_scl=1;transact(1,'h60,'h3107,0,2,0,0);hold_scl=0;
  #2000;if(!scl_oen || !sda_oen)$fatal(1,"I2C not released after timeout");
  transact(1,'h60,'h3108,0,0,0,'h6b);
  // Ready is asynchronous to camera_clk. A queued command must not touch
  // pins before a synchronized ready, nor duplicate when power returns.
  camera_power_ready=0;repeat(6)@(posedge camera_clk);
  begin
   integer before_starts;
   before_starts=starts;
   fork
    transact(1,'h60,'h3107,0,0,0,'hcd);
    begin
     repeat(20)@(negedge camera_clk);
     if(starts!=before_starts || camera_response_valid)$fatal(1,"I2C started before synchronized power ready");
     #57;camera_power_ready=1;
    end
   join
  end
  // Lost power during a real transaction returns exactly one held error,
  // releases the pins and allows a fresh request after power comes back.
  fork
   transact(1,'h60,'h3108,0,2,0,0);
   begin wait(dut.active);#57;camera_power_ready=0;end
  join
  if(!scl_oen || !sda_oen)$fatal(1,"Power-loss did not release pins");
  camera_power_ready=1;repeat(6)@(posedge camera_clk);
  transact(1,'h60,'h3108,0,0,0,'h6b);
  $display("PASS CAMERA_PIN_CDC tests=%0d writes=%0d reads=%0d",tests,writes,reads);$finish;
 end
 initial begin #20000000;$fatal(1,"watchdog");end
endmodule
