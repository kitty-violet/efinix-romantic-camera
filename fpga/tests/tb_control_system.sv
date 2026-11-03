`timescale 1ns/1ps
module tb_control_system #(parameter integer SERVICE_MHZ=150);
 reg reset=1,cpu_clk=0,service_clk=0,camera_clk=0;
 always #5 cpu_clk=~cpu_clk;always #(500.0/SERVICE_MHZ) service_clk=~service_clk;always #100 camera_clk=~camera_clk;
 reg board_ready=1,drained=1,external_fault=0,frame_event=0,native_external_write=0,camera_power_ready=1;
 reg event_valid=0;reg[5:0]event_bits=0;reg[7:0]event_target=0;wire event_ready;
 wire[3:0]flags;wire[7:0]run_state,fault;wire[15:0]style;
 wire view,prepared,camera_ok,bus_fault,cache_invalidated,warm_request;
 wire[9:0]frame_budget;wire[8:0]ui_value;wire[31:0]heartbeat,completed_sequence,software_irq_count,observer_pc;
 wire[63:0]camera_diagnostic;wire[31:0]isp_crc=0,isp_control=0,isp_written=0,isp_accepted=0;
 wire native_valid,native_write;wire[31:0]native_address,native_data;
 reg native_ready=0,native_response_valid=0,native_response_error=0;reg[31:0]native_response_data=0;
 wire flash_cs_n,flash_sclk,flash_mosi;reg flash_miso=0;
 wire scl_o,scl_oen,sda_o,sda_oen;tri1 scl_i,sda_i;reg slave_low=0;
 assign scl_i=scl_oen?1'bz:scl_o;assign sda_i=sda_oen?1'bz:sda_o;
 assign sda_i=slave_low?1'b0:1'bz;
 rv32_control_system #(.SPI_HALF_PERIOD(SERVICE_MHZ==200 ? 22:4),.BUS_TIMEOUT(SERVICE_MHZ==200 ? 5462:4096)) dut(.*);
 // Pin-level read-only SPI model. It must actually receive 03 + the bounded
 // address before returning payload bits; no bypass of the production engine.
 integer command_bits=0,payload_bit=0,spi_reads=0;reg[31:0]command_shift=0;
 always @(negedge flash_cs_n)begin command_bits=0;payload_bit=0;command_shift=0;flash_miso=0;spi_reads=spi_reads+1;end
 always @(posedge flash_sclk)if(!flash_cs_n && command_bits<32)begin
  command_shift={command_shift[30:0],flash_mosi};command_bits=command_bits+1;
  if(command_bits==32 && command_shift!=32'h03340000)$fatal(1,"SPI command/address %h",command_shift);
 end
 always @(negedge flash_sclk)if(!flash_cs_n && command_bits==32)begin
  flash_miso=((payload_bit/8)>>(7-payload_bit%8))&1;payload_bit=payload_bit+1;
 end
 // Pin-level sensor ID read, including repeated start and master last NACK.
 integer bit_count=0,byte_count=0,i2c_reads=0;reg active=0,reading=0,ack=0;
 reg[7:0]shift=0,send_byte=0;reg[15:0]camera_address=0;
 always @(negedge sda_i)if(scl_i===1 && !reset)begin active=1;bit_count=0;byte_count=0;reading=0;ack=0;end
 always @(posedge sda_i)if(scl_i===1 && !reset && active)active=0;
 always @(posedge scl_i)if(active && !reset)begin
  if(bit_count<8)begin
   shift={shift[6:0],sda_i};bit_count=bit_count+1;
   if(bit_count==8 && !reading)begin
    ack=byte_count!=0 ||(shift&8'hfe)==8'h60;
    if(byte_count==1)camera_address[15:8]=shift;
    if(byte_count==2)camera_address[7:0]=shift;
   end
  end else begin
   if(reading)begin if(sda_i!==1)$fatal(1,"I2C final NACK");reading=0;i2c_reads=i2c_reads+1;end
   else if(byte_count==0 && shift[0] && ack)begin
    if(camera_address!=16'h3107)$fatal(1,"I2C address");reading=1;send_byte=8'hcd;
   end
   bit_count=0;byte_count=byte_count+1;
  end
 end
 always @(negedge scl_i)begin
  if(!active || reset)slave_low=0;
  else if(reading)slave_low=bit_count<8?!send_byte[7-bit_count]:0;
  else slave_low=bit_count==8 && ack;
 end
 reg[31:0]memory[0:255];reg pending=0;integer delay_left=0,transactions=0,case_id=0,cycles=0;
 reg held_error=0;
 always @(posedge service_clk)begin
  if(reset)begin native_ready<=0;native_response_valid<=0;pending<=0;end
  else begin
   native_ready<=!pending && $urandom_range(0,3)!=0;native_response_valid<=0;
   if(native_valid && native_ready)begin
    if(native_address[31:24]!=8'he1)$fatal(1,"native escaped %h",native_address);
    pending<=1;delay_left<=$urandom_range(0,11);held_error<=case_id==1 && transactions==2;transactions<=transactions+1;
    native_response_data<=memory[native_address[9:2]];
    if(native_write)memory[native_address[9:2]]<=native_data;
   end
   if(pending)begin
    if(delay_left!=0)delay_left<=delay_left-1;
    else begin pending<=0;native_response_valid<=1;native_response_error<=held_error;end
   end
  end
 end
 always @(posedge cpu_clk)begin
  cycles<=cycles+1;
  if(case_id==0 && fault!=0)$fatal(1,"C integration fault %h PC=%h",fault,observer_pc);
  if(cycles>300000)$fatal(1,"watchdog PC=%h state=%h fault=%h",observer_pc,run_state,fault);
 end
 initial begin
  if(!$value$plusargs("CASE=%d",case_id))case_id=0;
  for(integer n=0;n<256;n=n+1)memory[n]=0;
  #37;reset=0;
  if(case_id==1)begin
   wait(fault!=0);repeat(20)@(posedge service_clk);
   if(fault!=8'hd1 || flags!=0 || prepared)$fatal(1,"fault failed containment");
   wait(flash_cs_n);$display("PASS RV32_REAL_BACKEND_FAULT transactions=%0d",transactions);$finish;
  end
  wait(run_state==8'ha5);repeat(20)@(posedge service_clk);
  if(!prepared || !camera_ok || flags!=3 || ui_value!=9'h111 || software_irq_count==0 ||
     spi_reads!=1 || i2c_reads!=1 || memory['hc0]!=32'h73b9e12a)$fatal(1,"incomplete integration");
  $display("PASS RV32_REAL_BACKEND_SPI_I2C_DMA_CDC cycles=%0d native=%0d IRQ=%0d",cycles,transactions,software_irq_count);$finish;
 end
endmodule
