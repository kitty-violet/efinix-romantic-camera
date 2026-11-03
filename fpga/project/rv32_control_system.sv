// Complete RV32/C control plane. All native port ownership stays in service_clk;
// only atomic requests cross from the CPU. I2C pins retain their 5MHz engine.
module rv32_control_system #(
 parameter integer SPI_HALF_PERIOD=16,BUS_TIMEOUT=4096
)(
 input wire reset,cpu_clk,service_clk,camera_clk,
 input wire board_ready,drained,external_fault,frame_event,native_external_write,camera_power_ready,
 input wire event_valid,output wire event_ready,input wire[5:0]event_bits,input wire[7:0]event_target,
 output wire[3:0]flags,output wire[7:0]run_state,fault,output wire[15:0]style,
 output wire view,prepared,camera_ok,bus_fault,cache_invalidated,warm_request,
 output wire[9:0]frame_budget,output wire[8:0]ui_value,
 output wire[31:0]heartbeat,completed_sequence,software_irq_count,observer_pc,
 output wire[63:0]camera_diagnostic,
 input wire[31:0]isp_crc,isp_control,isp_written,isp_accepted,
 output wire native_valid,native_write,output wire[31:0]native_address,native_data,
 input wire native_ready,native_response_valid,native_response_error,input wire[31:0]native_response_data,
 output wire flash_cs_n,flash_sclk,flash_mosi,input wire flash_miso,
 input wire scl_i,sda_i,output wire scl_o,scl_oen,sda_o,sda_oen
);
 wire psel,penable,pwrite,pready,pslverror;wire[15:0]paddr;wire[31:0]pwdata,prdata;
 wire protocol_error,interrupt_request,cpu_restart;
 assign observer_pc=32'd0;
 // Preserve a physical two-FF boundary before the vendor's internal BufferCC,
 // which the mapper may implement as an SRL. The vendor SoC remains unchanged.
 (* async_reg="true",syn_srlstyle="registers" *) reg[1:0]irq_sync;
 always @(posedge cpu_clk or posedge reset)
  if(reset)irq_sync<=0;else irq_sync<={irq_sync[0],interrupt_request};
 (* async_reg="true",syn_srlstyle="registers" *)reg[1:0]cpu_fault_sync,camera_power_sync;
 always @(posedge service_clk or posedge reset)
  if(reset)begin cpu_fault_sync<=0;camera_power_sync<=0;end
  else begin cpu_fault_sync<={cpu_fault_sync[0],protocol_error};camera_power_sync<={camera_power_sync[0],camera_power_ready};end
 sapphire_candidate cpu(.clk(cpu_clk),.reset(reset),.interrupt_request(irq_sync[1]),
  .system_reset(cpu_restart),.*);
 wire request_valid,request_ready,request_write,response_valid,response_ready,response_error;
 wire[15:0]request_address;wire[31:0]request_data,response_data;
 control_rpc_bridge cpu_service(.reset(reset),.*);
 wire spi_start,spi_start_ready,spi_busy,spi_done,spi_error,spi_valid,spi_ready;
 wire[23:0]spi_address,spi_length;wire[31:0]spi_data;
 wire camera_request_valid,camera_request_ready,camera_response_valid,camera_response_ready;
 wire[32:0]camera_request;wire[3:0]camera_response_error;
 wire[2:0]camera_response_error_byte;wire[7:0]camera_response_data;
 rv32_service_backend #(.BUS_TIMEOUT(BUS_TIMEOUT)) service(.clk(service_clk),
  .external_fault(external_fault || cpu_fault_sync[1]),.camera_power_ready(camera_power_sync[1]),.*);
 camera_service_bridge camera(.*);
 (* async_reg="true",syn_srlstyle="registers" *)reg miso_meta,miso_sync;
 always @(posedge service_clk)begin
  miso_meta<=flash_miso;miso_sync<=miso_meta;
 end
 spi_flash_read207 #(.HALF_PERIOD(SPI_HALF_PERIOD)) flash(
  .clk(service_clk),.reset(reset),.start(spi_start),.address(spi_address),.byte_count(spi_length),
  .start_ready(spi_start_ready),.busy(spi_busy),.done(spi_done),.error(spi_error),
  .cs_n(flash_cs_n),.sclk(flash_sclk),.mosi(flash_mosi),.miso(miso_sync),
  .data(spi_data),.valid(spi_valid),.ready(spi_ready));
endmodule
