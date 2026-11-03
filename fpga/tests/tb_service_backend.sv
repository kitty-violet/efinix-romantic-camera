`timescale 1ns/1ps
module tb_service_backend;
 wire cpu_restart=0;
 reg clk=0,cpu_clk=0,reset=1;
 always #3.333 clk=~clk;always #5 cpu_clk=~cpu_clk;
 reg psel=0,penable=0,pwrite=0;reg[15:0]paddr=0;reg[31:0]pwdata=0;
 wire pready,pslverror,protocol_error;wire[31:0]prdata;
 wire request_valid,request_ready,request_write,response_valid,response_ready,response_error;
 wire[15:0]request_address;wire[31:0]request_data,response_data;
 control_rpc_bridge bridge(.service_clk(clk),.*);
 reg board_ready=1,drained=1,external_fault=0,event_valid=0,frame_event=0;
 wire interrupt_request;wire[8:0]ui_value;wire[31:0]software_irq_count;
 reg native_external_write=0;wire cache_invalidated,warm_request;wire[9:0]frame_budget;wire[63:0]camera_diagnostic;
 reg camera_power_ready=1;
 reg[5:0]event_bits=0;reg[7:0]event_target=0;
 wire event_ready;wire[3:0]flags;wire[7:0]run_state,fault;wire[15:0]style;
 wire view,prepared,camera_ok,bus_fault;wire[31:0]heartbeat,completed_sequence;
 wire[31:0]isp_crc=32'h11112222,isp_control=32'h12341700,isp_written=32'h56aabbcc,isp_accepted=32'h33334444;
 wire native_valid,native_write;wire[31:0]native_address,native_data;
 reg native_ready=0,native_response_valid=0,native_response_error=0;
 reg[31:0]native_response_data=0;
 wire spi_start,spi_ready;wire[23:0]spi_address,spi_length;
 reg spi_busy=0,spi_done=0,spi_error=0,spi_valid=0;wire spi_start_ready=!spi_busy;
 reg[31:0]spi_data=0;integer spi_words=0,spi_accepted=0,spi_started=0;
 wire camera_request_valid,camera_response_ready;
 reg camera_response_valid=0;wire camera_request_ready=!camera_response_valid;wire[32:0]camera_request;
 reg[3:0]camera_response_error=0;
 reg[2:0]camera_response_error_byte=0;reg[7:0]camera_response_data=0;
 rv32_service_backend #(.BUS_TIMEOUT(64)) dut(.*);
 reg[31:0]memory[0:255];reg native_pending=0,inject_error=0,block_native=0;
 integer native_delay=0,native_count=0,native_write_count=0,camera_count=0;
 reg held_native_error;
 always @(posedge clk)begin
  if(reset)begin
   native_ready<=0;native_response_valid<=0;native_pending<=0;native_delay<=0;
   spi_busy<=0;spi_done<=0;spi_error<=0;spi_valid<=0;
   camera_response_valid<=0;
  end else begin
   native_response_valid<=0;native_ready<=!block_native && !native_pending && $urandom_range(0,3)!=0;
   if(native_valid && native_ready)begin
    native_count<=native_count+1;native_pending<=1;native_delay<=$urandom_range(0,10);
    held_native_error<=inject_error;
    if(native_write)begin memory[native_address[9:2]]<=native_data;native_write_count<=native_write_count+1;end
    native_response_data<=memory[native_address[9:2]];
   end
   if(native_pending)begin
    if(native_delay!=0)native_delay<=native_delay-1;
    else begin native_pending<=0;native_response_valid<=1;native_response_error<=held_native_error;end
   end
   spi_done<=0;
   if(spi_start && spi_start_ready)begin
    spi_busy<=1;spi_words<=spi_length/4;spi_data<=32'habcdef01;spi_started<=spi_started+1;
   end
   if(spi_busy && !spi_valid && $urandom_range(0,3)!=0)spi_valid<=1;
   if(spi_valid && spi_ready)begin
    spi_valid<=0;spi_words<=spi_words-1;spi_data<=spi_data+1;spi_accepted<=spi_accepted+1;
    if(spi_words==1)begin spi_busy<=0;spi_done<=1;end
   end
   if(camera_request_valid && camera_request_ready)begin
    camera_count<=camera_count+1;camera_response_valid<=1;
    camera_response_error<=0;camera_response_error_byte<=0;
    if(camera_request!==33'h131076000)$fatal(1,"camera atomic payload %h",camera_request);
    camera_response_data<=8'hcd;
   end
   if(camera_response_valid && camera_response_ready)camera_response_valid<=0;
  end
 end
 reg stalled=0;reg[64:0]held;
 always @(posedge clk)begin
  if(reset)stalled<=0;
  else begin
   if(stalled && (!native_valid || held!=={native_write,native_address,native_data}))$fatal(1,"native offer changed before accept");
   stalled<=native_valid && !native_ready;held<={native_write,native_address,native_data};
  end
 end
 task automatic access(input bit wr,input[15:0]addr,input[31:0]data,output[31:0]result,input bit expect_error=0);
  integer age;
  begin
   @(negedge cpu_clk);psel=1;penable=0;pwrite=wr;paddr=addr;pwdata=data;
   @(negedge cpu_clk);penable=1;age=0;
   do begin @(posedge cpu_clk);age=age+1;if(age>500)$fatal(1,"APB timeout");end while(!pready);
   result=prdata;if(pslverror!==expect_error)$fatal(1,"APB error addr=%h wr=%b",addr,wr);
   @(negedge cpu_clk);psel=0;penable=0;
  end
 endtask
 reg[31:0]r,tmp;integer before_count;
 task automatic wr(input[15:0]addr,input[31:0]v);begin access(1,addr,v,tmp);end endtask
 task automatic rd(input[15:0]addr);begin access(0,addr,0,r);end endtask
 task automatic wait_done(input[15:0]addr,input bit expect_fault=0);
  begin
   for(integer age=0;age<100;age=age+1)begin
    rd(addr);if(r[2] && !expect_fault)$fatal(1,"wrong IO fault %h %h",addr,r);
    if(r[1])begin if(r[2]!==expect_fault)$fatal(1,"missing IO fault");return;end
   end
   $fatal(1,"operation timeout %h",addr);
  end
 endtask
 task automatic restart;
  begin @(negedge cpu_clk);reset=1;psel=0;penable=0;block_native=0;inject_error=0;event_valid=0;
   #43;reset=0;repeat(20)@(posedge cpu_clk);
  end
 endtask
 task automatic enqueue(input[5:0]ev,input[7:0]target);
  begin @(negedge clk);event_valid=1;event_bits=ev;event_target=target;
   do @(posedge clk);while(!event_ready);
   @(negedge clk);event_valid=0;
  end
 endtask
 initial begin
  for(integer k=0;k<256;k=k+1)memory[k]=k*32'h1020304;
  #13;reset=0;
  wr('h50,32'h31076000);wr('h54,2);wait_done('h58);rd('h5c);if(r!='hcd)$fatal(1,"camera response");
  if(camera_diagnostic[63:60]!==4'd1 || camera_diagnostic[15:8]!==8'hcd)
   $fatal(1,"camera ID-high diagnostic phase/value %h",camera_diagnostic);
  wr('h60,1);wr('h90,1);wr('h64,3);if(flags!=3)$fatal(1,"run permission");
  wr('h10,32'he1000000);wr('h14,32'h12345678);wr('h18,2);wait_done('h1c);
  wr('h18,1);wait_done('h1c);rd('h20);if(r!=32'h12345678)$fatal(1,"manual response");
  // The reserved DMA must prevent software from stealing its native/SPI port.
  wr('h38,32'h340000);wr('h3c,32'he1000000);wr('h40,64);wr('h44,1);
  access(1,'h18,1,tmp,1);wait_done('h48);
  if(!bus_fault || prepared || flags!=0)$fatal(1,"Rejected access did not revoke permissions");
  for(integer k=0;k<16;k=k+1)if(memory[k]!==32'habcdef01+k)$fatal(1,"SPI DMA data %d",k);
  rd('h4c);if(r!=32'h9b354c27)$fatal(1,"SPI DMA CRC %h",r);
  restart();
  wr('h38,32'he1000000);wr('h3c,32'he1000100);wr('h40,64);wr('h44,2);wait_done('h48);
  for(integer k=0;k<16;k=k+1)if(memory[64+k]!==memory[k])$fatal(1,"copy");
  wr('h38,32'he1000100);wr('h44,3);wait_done('h48);rd('h4c);if(r!=32'h9b354c27)$fatal(1,"DDR CRC");
  wr('h24,32'h340000);wr('h28,16);wr('h2c,1);
  for(integer k=0;k<4;k=k+1)begin
   do rd('h30);while(!r[3]);rd('h34);if(r!==32'habcdef01+k)$fatal(1,"manual SPI");
  end
  wait_done('h30);
  // Queue several distinct commands while C is occupied, preserving order/seq.
  for(integer k=0;k<4;k=k+1)enqueue(1<<k,k);
  if(event_ready)$fatal(1,"queue capacity");
  for(integer k=0;k<4;k=k+1)begin
   rd('h08);if(r!=1<<k)$fatal(1,"event order");
   rd('h94);if(r!=k)$fatal(1,"target atomicity");
   rd('h98);if(r!=k+1)$fatal(1,"sequence");
   wr('h0c,1<<k);wr('h9c,k+1);
  end
  rd('h08);if(r!=0 || completed_sequence!=4)$fatal(1,"queue drain");
  rd('h88);if(r!=isp_written)$fatal(1,"ISP snapshot");
  wr('h64,5);
  @(negedge clk);native_external_write=1;
  @(negedge clk);native_external_write=0;
  if(prepared || flags!=4 || !cache_invalidated)$fatal(1,"External upload permission revocation");
  access(1,'h64,3,tmp,1);
  wr('h6c,7);if(style!=7)$fatal(1,"Manual slot7 bypass");
  access(1,'h6c,0,tmp,1);access(1,'h6c,16'h0607,tmp,1);
  wr('hb0,1);if(cache_invalidated)$fatal(1,"Cache invalidation clear");
  wr('hb8,3);if(frame_budget!=3)$fatal(1,"Finite frame budget");
  // Timeout before acceptance: do not withdraw or mutate the offer. Revoke run
  // immediately, then accept/drain the late response with no follow-up command.
  restart();wr('h10,32'he1000000);
  block_native=1;repeat(5)@(posedge clk);wr('h18,1);repeat(100)@(posedge clk);
  if(!native_valid || !bus_fault || flags!=0 || prepared)$fatal(1,"timeout containment");
  before_count=native_count;block_native=0;wait_done('h1c,1);
  repeat(100)@(posedge clk);if(native_count!=before_count+1)$fatal(1,"late response drain");
  restart();
  // A native error during SPI copy drains only SPI, with no more native writes.
  inject_error=1;before_count=native_write_count;
  wr('h38,32'h340000);wr('h3c,32'he1000000);wr('h40,64);wr('h44,1);wait_done('h48,1);
  wait(!spi_busy);repeat(100)@(posedge clk);
  if(native_write_count!=before_count+1 || !bus_fault || protocol_error)$fatal(1,"fault SPI drain");
  $display("PASS BACKEND_CDC_DMA_OWNERSHIP native=%0d SPI_words=%0d camera=%0d",native_count,spi_accepted,camera_count);$finish;
 end
 initial begin #2000000;$fatal(1,"watchdog");end
endmodule
