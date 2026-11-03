// CPU control registers and one-owner native/SPI arbitration, all at service_clk.
// No per-word CPU polling during bulk loads. Native writes complete ONLY when
// their actual owner-qualified response arrives, not at request acceptance.
module rv32_service_backend #(parameter integer BUS_TIMEOUT=4096)(
 input wire clk,reset,
 input wire request_valid,output wire request_ready,input wire request_write,
 input wire[15:0]request_address,input wire[31:0]request_data,
 output reg response_valid,input wire response_ready,
 output reg response_error,output reg[31:0]response_data,
 input wire board_ready,drained,external_fault,frame_event,native_external_write,camera_power_ready,
 output wire interrupt_request,output reg[8:0]ui_value,output wire[31:0]software_irq_count,
 output reg cache_invalidated,warm_request,output reg[9:0]frame_budget,
 output wire[63:0]camera_diagnostic,
 input wire event_valid,output wire event_ready,
 input wire[5:0]event_bits,input wire[7:0]event_target,
 output reg[3:0]flags,output reg[7:0]run_state,fault,
 output reg[15:0]style,output reg view,prepared,camera_ok,
 output wire[31:0]heartbeat,completed_sequence,
 output wire bus_fault,
 input wire[31:0]isp_crc,isp_control,isp_written,isp_accepted,
 output wire native_valid,native_write,output wire[31:0]native_address,native_data,
 input wire native_ready,native_response_valid,native_response_error,
 input wire[31:0]native_response_data,
 output wire spi_start,input wire spi_start_ready,
 output wire[23:0]spi_address,spi_length,
 input wire spi_busy,spi_done,spi_error,spi_valid,input wire[31:0]spi_data,
 output wire spi_ready,
 output wire camera_request_valid,input wire camera_request_ready,
 output wire[32:0]camera_request,
 input wire camera_response_valid,output wire camera_response_ready,
 input wire[3:0]camera_response_error,input wire[2:0]camera_response_error_byte,
 input wire[7:0]camera_response_data
);
 wire[31:0]dma_source,dma_target,dma_length;
 reg[2:0]dma_operation;
 reg dma_start,dma_reserved,bulk_interrupt;
 wire dma_start_ready,dma_busy,dma_done,dma_error,dma_poison;
 wire[7:0]dma_fault;wire[31:0]dma_crc,dma_word_result;
 wire dma_valid,dma_write,dma_ready,dma_response_valid;
 wire[31:0]dma_address,dma_data;
 wire dma_spi_start,dma_spi_start_ready,dma_spi_valid,dma_spi_ready;
 wire[23:0]dma_spi_address,dma_spi_length;
 wire[23:0]manual_spi_address,manual_spi_length;
 reg manual_spi_queued,manual_spi_busy,manual_spi_done,manual_spi_error,spi_dma_owner;
 wire[31:0]camera_word;reg[31:0]camera_result;
 reg camera_queued,camera_busy,camera_done,camera_error,camera_read;
 reg[7:0]camera_index,camera_last_actual;
 reg[3:0]camera_last_error;
 reg[15:0]camera_chip_id;
 wire[3:0]camera_phase=camera_ok ? 4'd15 : camera_read ?
  (camera_word[31:16]==16'h3107 ? 4'd1 : camera_word[31:16]==16'h3108 ? 4'd2 : 4'd4) : 4'd3;
 assign camera_diagnostic={camera_phase,camera_last_error,camera_index,camera_word[31:16],
  camera_last_actual,camera_word[7:0],camera_chip_id};
 reg unexpected_native,service_access_fault;
 reg dma_done_previous,dma_interrupt_pending,frame_interrupt_pending;
 reg[5:0]event_queue[0:3];reg[7:0]target_queue[0:3];
 reg[1:0]event_read,event_write;reg[2:0]event_count;
 reg[31:0]dispatched_sequence;
 // FIFO order: head=last dispatched+1; next=head+count. Only empty next is read.
 wire[31:0]head_sequence=dispatched_sequence+32'd1;
 wire[31:0]next_sequence=head_sequence;
 wire fire=request_valid && request_ready;
 wire consume=fire && request_write && request_address==16'h000c && event_count!=0 &&
              request_data=={26'd0,event_queue[event_read]};
 wire push=event_valid && event_ready;
 assign event_ready=event_count<4 && fault==0;
 // Do not fold response_ready back into request_ready: the RPC bridge can
 // accept an immediate reply, so response_ready itself may depend on the
 // accepted request. Retire the registered reply before accepting another
 // register access. This removes the combinational ready/accepted loop.
 assign request_ready=!response_valid;
 control_dsp_register #(.WIDTH(32)) dma_source_storage(.clk(clk),.reset(reset),
  .enable(fire && request_write && ((request_address==16'h10 || request_address==16'h38) && !dma_reserved)),.data(request_data),.q(dma_source));
 control_dsp_register #(.WIDTH(32)) dma_target_storage(.clk(clk),.reset(reset),
  .enable(fire && request_write && ((request_address==16'h14 || request_address==16'h3c) && !dma_reserved)),.data(request_data),.q(dma_target));
 control_dsp_register #(.WIDTH(32)) dma_length_storage(.clk(clk),.reset(reset),
  .enable(fire && request_write && (request_address==16'h40 && !dma_reserved)),.data(request_data),.q(dma_length));
 control_dsp_register #(.WIDTH(24)) manual_spi_address_storage(.clk(clk),.reset(reset),
  .enable(fire && request_write && (request_address==16'h24 && !manual_spi_busy && request_data[31:24]==0)),.data(request_data[23:0]),.q(manual_spi_address));
 control_dsp_register #(.WIDTH(24)) manual_spi_length_storage(.clk(clk),.reset(reset),
  .enable(fire && request_write && (request_address==16'h28 && !manual_spi_busy && request_data[31:24]==0)),.data(request_data[23:0]),.q(manual_spi_length));
 control_dsp_register #(.WIDTH(32)) camera_word_storage(.clk(clk),.reset(reset),
  .enable(fire && request_write && (request_address==16'h50 && !camera_busy)),.data(request_data),.q(camera_word));
 control_dsp_register #(.WIDTH(32)) software_irq_count_storage(.clk(clk),.reset(reset),
  .enable(fire && request_write && (request_address==16'hac)),.data(request_data),.q(software_irq_count));
 control_dsp_register #(.WIDTH(32)) completed_sequence_storage(.clk(clk),.reset(reset),
  .enable(fire && request_write && (request_address==16'h9c && request_data==dispatched_sequence)),.data(request_data),.q(completed_sequence));
 control_dsp_counter #(.WIDTH(32)) heartbeat_storage(.clk(clk),.reset(reset),
  .enable(fire && request_write && request_address==16'h74),.q(heartbeat));
 wire native_owner=dma_reserved;
 wire native_capture_ready,native_register_fault,registered_response_valid,registered_response_error;
 wire[31:0]registered_response_data;
 native_command_register native_register(.clk(clk),.reset(reset),
  .s_valid(dma_reserved && dma_valid),.s_ready(native_capture_ready),.s_write(dma_write),.s_address(dma_address),.s_data(dma_data),
  .s_response_valid(registered_response_valid),.s_response_error(registered_response_error),.s_response_data(registered_response_data),
  .m_valid(native_valid),.m_write(native_write),.m_address(native_address),.m_data(native_data),
  .m_ready(native_ready),.m_response_valid(native_response_valid),.m_response_error(native_response_error),.m_response_data(native_response_data),
  .protocol_fault(native_register_fault));
 assign dma_ready=native_capture_ready && dma_reserved;
 assign dma_response_valid=registered_response_valid && dma_reserved;
 assign bus_fault=external_fault || unexpected_native || dma_error || dma_poison || native_register_fault || service_access_fault;
 assign interrupt_request=dma_interrupt_pending || frame_interrupt_pending || event_count!=0 || bus_fault;
 assign spi_start=!reset && (manual_spi_queued || dma_spi_start);
 assign spi_address=manual_spi_queued?manual_spi_address:dma_spi_address;
 assign spi_length=manual_spi_queued?manual_spi_length:dma_spi_length;
 assign dma_spi_start_ready=spi_start_ready && !manual_spi_queued && !manual_spi_busy;
 assign dma_spi_valid=spi_valid && spi_dma_owner;
 // After a DMA failure, finish its READ-ONLY SPI transaction without performing
 // further native writes. Do not reset a shared physical bus behind its owner.
 assign spi_ready=spi_dma_owner?(dma_error || dma_spi_ready):
   (fire && !request_write && request_address==16'h0034 && manual_spi_busy && spi_valid);
 assign camera_request_valid=camera_queued && !reset;
 assign camera_request={camera_read,camera_word};
 assign camera_response_ready=camera_busy && !camera_queued;
 service_word_dma #(.BUS_TIMEOUT(BUS_TIMEOUT)) dma(
  .clk(clk),.reset(reset),.start(dma_start),.operation(dma_operation),
  .source(dma_source),.target(dma_target),.byte_count(dma_operation>=5?32'd4:dma_length),
  .start_ready(dma_start_ready),.busy(dma_busy),.done(dma_done),.error(dma_error),
  .poisoned(dma_poison),.fault(dma_fault),.crc_result(dma_crc),.word_result(dma_word_result),
  .cmd_valid(dma_valid),.cmd_write(dma_write),.cmd_address(dma_address),.cmd_data(dma_data),
  .cmd_ready(dma_ready),.rsp_valid(dma_response_valid),.rsp_error(registered_response_error),.rsp_data(registered_response_data),
  .spi_start(dma_spi_start),.spi_start_ready(dma_spi_start_ready),
  .spi_address(dma_spi_address),.spi_length(dma_spi_length),
  .spi_valid(dma_spi_valid),.spi_error(spi_error && spi_dma_owner),.spi_data(spi_data),.spi_ready(dma_spi_ready));
 wire manual_address_legal=(dma_source[31:24]==8'he1) ||
  (dma_source>=32'hf8100000 && dma_source<32'hf8100080) ||
  (dma_source>=32'hf9000f00 && dma_source<32'hf9000f40);
 always @(posedge clk)begin
  if(reset)begin
   response_valid<=0;response_error<=0;response_data<=0;
   flags<=4;run_state<=0;fault<=0;style<=0;view<=0;prepared<=0;camera_ok<=0;
   
   dma_operation<=0;dma_start<=0;dma_reserved<=0;bulk_interrupt<=0;
   manual_spi_queued<=0;manual_spi_busy<=0;
   manual_spi_done<=0;manual_spi_error<=0;spi_dma_owner<=0;
   camera_result<=0;camera_queued<=0;camera_busy<=0;camera_done<=0;camera_error<=0;camera_read<=0;
   camera_index<=0;camera_last_actual<=0;camera_last_error<=0;camera_chip_id<=0;
   cache_invalidated<=0;warm_request<=0;frame_budget<=1023;
   event_read<=0;event_write<=0;event_count<=0;dispatched_sequence<=0;
   unexpected_native<=0;service_access_fault<=0;dma_done_previous<=0;dma_interrupt_pending<=0;frame_interrupt_pending<=0;
   ui_value<=0;
  end else begin
   dma_start<=0;
   // Sapphire Lite may not raise an exception for PSLVERROR. A rejected
   // service access must therefore revoke permission in hardware as well.
   if(response_valid && response_error)service_access_fault<=1;
   dma_done_previous<=dma_done;
   if(response_valid && response_ready)response_valid<=0;
   if(push)begin
    event_queue[event_write]<=event_bits;target_queue[event_write]<=event_target;
    event_write<=event_write+1'b1;
   end
   if(consume)begin
    event_read<=event_read+1'b1;dispatched_sequence<=head_sequence;
   end
   case({push,consume})2'b10:event_count<=event_count+1'b1;2'b01:event_count<=event_count-1'b1;default:begin end endcase
   if(native_response_valid && !native_owner)unexpected_native<=1;
   if(dma_reserved && dma_done && !dma_busy && !dma_start)dma_reserved<=0;
   if(spi_start && spi_start_ready)begin
    spi_dma_owner<=!manual_spi_queued;
    if(manual_spi_queued)manual_spi_queued<=0;
   end
   if(manual_spi_busy && !manual_spi_queued && !spi_dma_owner && (spi_done || spi_error))begin
    manual_spi_busy<=0;manual_spi_done<=1;manual_spi_error<=spi_error;
   end
   if(camera_request_valid && camera_request_ready)camera_queued<=0;
   if(camera_response_valid && camera_response_ready)begin
    camera_busy<=0;camera_done<=1;camera_error<=camera_response_error!=0;
    camera_result<=camera_response_error!=0?{25'd0,camera_response_error_byte,camera_response_error}:{24'd0,camera_response_data};
    camera_last_actual<=camera_response_error!=0?{5'd0,camera_response_error_byte}:camera_response_data;
    camera_last_error<=camera_response_error;
    if(camera_response_error==0)begin
     if(!camera_read)camera_index<=camera_index+1'b1;
     else if(camera_word[31:16]==16'h3107)camera_chip_id[15:8]<=camera_response_data;
     else if(camera_word[31:16]==16'h3108)camera_chip_id[7:0]<=camera_response_data;
    end
   end
   if(fire)begin
    response_valid<=1;response_error<=0;response_data<=0;
    if(request_address[1:0]!=0)response_error<=1;
    else if(!request_write)case(request_address)
     'h00:response_data<=32'h52563233;
     'h04:response_data<={28'd0,camera_power_ready,bus_fault,drained,board_ready && camera_ok};
     'h08:response_data<=event_count!=0?{26'd0,event_queue[event_read]}:0;
     'h10:response_data<=dma_source;
     'h14:response_data<=dma_target;
     'h1c:response_data<={29'd0,dma_error,dma_done,dma_reserved};
     'h20:response_data<=dma_word_result;
     'h30:response_data<={28'd0,spi_valid && !spi_dma_owner && manual_spi_busy,manual_spi_error,manual_spi_done,manual_spi_busy};
     'h34:begin response_data<=spi_data;if(!(spi_valid && !spi_dma_owner && manual_spi_busy))response_error<=1;end
     'h48:response_data<={16'd0,dma_fault,5'd0,dma_error,dma_done,dma_reserved};
     'h4c:response_data<=dma_crc;
     'h58:response_data<={29'd0,camera_error,camera_done,camera_busy};
     'h5c:response_data<=camera_result;
     'h60:response_data<={31'd0,camera_ok};
     'h64:response_data<={28'd0,flags};
     'h68:response_data<={24'd0,run_state};
     'h6c:response_data<={16'd0,style};
     'h70:response_data<={24'd0,fault};
     'h74:response_data<=heartbeat;
     'h7c:response_data<={31'd0,view};
     'h80:response_data<=isp_crc;
     'h84:response_data<=isp_control;
     'h88:response_data<=isp_written;
     'h8c:response_data<=isp_accepted;
     'h90:response_data<={31'd0,prepared};
     'h94:response_data<=event_count!=0?{24'd0,target_queue[event_read]}:0;
     'h98:response_data<=event_count!=0?head_sequence:next_sequence;
     'h9c:response_data<=completed_sequence;
     'ha0:response_data<={23'd0,ui_value};
     'ha4:response_data<={28'd0,bus_fault,event_count!=0,frame_interrupt_pending,dma_interrupt_pending};
     'hac:response_data<=software_irq_count;
     'hb0:response_data<={31'd0,cache_invalidated};
     'hb4:response_data<={31'd0,warm_request};
     'hb8:response_data<={22'd0,frame_budget};
     default:response_error<=1;
    endcase
    else case(request_address)
     'h0c:if(!consume)response_error<=1;
     'h10:if(dma_reserved)response_error<=1;
     'h14:if(dma_reserved)response_error<=1;
     'h18:begin
      if(native_owner || bus_fault || !dma_start_ready || !manual_address_legal || dma_source[1:0]!=0 ||
         (request_data!=1 && request_data!=2))response_error<=1;
      else begin dma_operation<=request_data==2?6:5;dma_start<=1;dma_reserved<=1;bulk_interrupt<=0;end
     end
     'h24:if(manual_spi_busy || request_data[31:24]!=0)response_error<=1;
     'h28:if(manual_spi_busy || request_data[31:24]!=0)response_error<=1;
     'h2c:begin
      if(request_data!=1 || manual_spi_busy || dma_reserved || spi_busy || !spi_start_ready ||
         manual_spi_address<24'h340000 || manual_spi_length==0 || manual_spi_length[1:0]!=0 ||
         {1'b0,manual_spi_address}+{1'b0,manual_spi_length}>25'h800000)response_error<=1;
      else begin manual_spi_queued<=1;manual_spi_busy<=1;manual_spi_done<=0;manual_spi_error<=0;end
     end
     'h38:if(dma_reserved)response_error<=1;
     'h3c:if(dma_reserved)response_error<=1;
     'h40:if(dma_reserved)response_error<=1;
     'h44:begin
      if(native_owner || bus_fault || manual_spi_busy || spi_busy || !dma_start_ready ||
         request_data<1 || request_data>4)response_error<=1;
      else begin dma_operation<=request_data[2:0];dma_start<=1;dma_reserved<=1;bulk_interrupt<=1;end
     end
     'h50:if(camera_busy)response_error<=1;
     'h54:begin
      if(camera_busy || (request_data!=1 && request_data!=2))response_error<=1;
      else begin camera_busy<=1;camera_queued<=1;camera_done<=0;camera_error<=0;camera_read<=request_data==2;end
     end
     'h60:if(request_data>1 || camera_busy || camera_error)response_error<=1;else camera_ok<=request_data[0];
     'h64:if(request_data>15 || (request_data[1] && (!prepared || fault!=0 || bus_fault || !board_ready ||
               !request_data[0] || request_data[2] || request_data[3])))response_error<=1;else flags<=request_data[3:0];
     'h68:run_state<=request_data[7:0];
     'h6c:if(request_data!=7 && (request_data[31:16]!=0 || request_data[15:8]<1 || request_data[15:8]>6 ||
             request_data[7:0]>=request_data[15:8]))response_error<=1;else style<=request_data[15:0];
     'h70:begin if(request_data!=0 && fault==0)fault<=request_data[7:0];end
     'h74:begin end
     'h7c:if(request_data>1)response_error<=1;else view<=request_data[0];
     'h90:if(request_data>1 || (request_data[0] && (fault!=0 || bus_fault || !camera_ok)))response_error<=1;else prepared<=request_data[0];
     'h9c:if(request_data!=dispatched_sequence)response_error<=1;
     'ha0:if(request_data>511)response_error<=1;else ui_value<=request_data[8:0];
     'ha8:begin
      if(request_data>3)response_error<=1;
      else begin
       if(request_data[0])dma_interrupt_pending<=0;
       if(request_data[1])frame_interrupt_pending<=0;
      end
     end
     'hac:begin end
     'hb0:if(request_data!=1 || flags[1] || flags[3])response_error<=1;else cache_invalidated<=0;
     'hb4:if(request_data>1 || bus_fault || fault!=0)response_error<=1;else warm_request<=request_data[0];
     'hb8:if(request_data==0 || request_data>1023 || flags[1])response_error<=1;else frame_budget<=request_data[9:0];
     default:response_error<=1;
    endcase
   end
   // Sticky permission revocation is independent of CPU polling latency.
   if(bus_fault || fault!=0)begin prepared<=0;flags<=0;end
   // A completion coincident with W1C remains pending, so it cannot be lost.
   if(dma_done && !dma_done_previous && bulk_interrupt)dma_interrupt_pending<=1;
   if(frame_event)frame_interrupt_pending<=1;
   if(native_external_write)begin
    cache_invalidated<=1;flags[0]<=0;prepared<=0;
    if(flags[1] || flags[3])begin fault<=8'h09;flags<=0;prepared<=0;end
   end
  end
 end
endmodule
