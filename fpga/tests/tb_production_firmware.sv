`timescale 1ns/1ps
// Executes the real generated RV32 core and C image. Peripheral models are
// transaction-level contract checks, not a whole-board or DDR simulation.
module tb_firmware;
 reg clk=0,reset=1;
 always #5 clk=~clk;
 wire[15:0] address;wire[31:0]wdata;wire sel,enable,writing;reg[31:0]rdata;
 wire sys_reset;
 wire[31:0]cpu_pc;wire control_irq;
 assign cpu_pc=dut.soc.system_cores_0_logic_cpu_iBus_cmd_payload_pc;
 integer cycles=0;
 wire apb_ready=(cycles%7)!=2 && (cycles%7)!=3;
 sapphire_candidate dut(.clk(clk),.reset(reset),.interrupt_request(control_irq),.system_reset(sys_reset),
 .paddr(address),.pwdata(wdata),.psel(sel),.penable(enable),.pwrite(writing),
 .pready(apb_ready),.pslverror(1'b0),.prdata(rdata));
 reg[31:0]directory[0:143],camera_table[0:163];reg[7:0]isp[0:767],staging[0:767];
 reg[7:0]camera_registers[0:65535];
 reg[31:0]native_address=0,native_data=0,native_result=0,native_status=2;
 reg[31:0]spi_address=0,spi_length=0;integer spi_index=0;
 reg[31:0]dma_source=0,dma_target=0,dma_length=0,dma_crc=0;
 reg[31:0]camera_word=0,camera_result=0,camera_status=2;
 reg[31:0]flags=0,run_state=0,style=0,ready=0,fault=0,view=0;
 reg[31:0]events=0,request_seq=0,completed_seq=0,frame_count=0;
 reg irq_pending=0;reg[31:0]ui_value=0,irq_handled=0;integer ui_updates=0;
 assign control_irq=irq_pending || events!=0;
 reg[31:0]native_control=0,isp_expected=0,isp_accepted=0,isp_peek=0,isp_control=0;
 reg[31:0]active_compute_crc=0,active_coeff_crc=0;
 reg[31:0]host_command=0,manual_frames=3,frame_budget=1023;
 reg cache_invalidated=0;
 integer camera_writes=0,dma_commands=0,native_commands=0,case_id=0,before_frame;
 reg configured=0;wire fire=sel&&enable&&apb_ready;
 reg[31:0]temp;integer index;
 function automatic[31:0]native_read(input[31:0]a);
  if(a>=32'he1300000 && a<32'he1300300)begin
   native_read={isp[a-32'he1300000+3],isp[a-32'he1300000+2],isp[a-32'he1300000+1],isp[a-32'he1300000]};
  end else case(a)
   32'hf8100004:native_read=1;
   32'hf8100008:native_read=native_control;
   32'hf810005c:native_read=frame_count;
   32'hf9000f00:native_read=host_command;
   32'hf9000f08:native_read=1;
   32'hf9000f0c:native_read=active_coeff_crc;
   32'hf9000f10:native_read=active_compute_crc;
   32'hf9000f14:native_read=directory[9];
   32'hf9000f18:native_read=manual_frames;
   default:native_read=0;
  endcase
 endfunction
 function automatic[31:0]memory_crc(input[31:0]a,input[31:0]len);
  integer n;reg found;
  begin
   found=0;memory_crc=0;
   if(a==32'h341000 || a==32'he1100000)begin
    if(len!=132432)$fatal(1,"DMA length");memory_crc=directory[9];found=1;
   end
   if(a==32'h362000 || a==32'he1300000)begin
    if(len!=768)$fatal(1,"ISP length");memory_crc=directory[12];found=1;
   end
   for(n=0;n<6;n=n+1)begin
    if(a==directory[18+16*n] || a==32'he1800000+(n<<20))begin
     if(len!=418944)$fatal(1,"Compute length");memory_crc=directory[20+16*n];found=1;
    end
    if(a==directory[21+16*n] || a==32'he18c0000+(n<<20))begin
     if(len!=20480)$fatal(1,"Weights length");memory_crc=directory[23+16*n];found=1;
    end
   end
   if(a==32'he1000000)begin memory_crc=active_compute_crc;found=1;end
   if(a==32'he1200000)begin memory_crc=active_coeff_crc;found=1;end
   if(!found)$fatal(1,"Unexpected DMA source %h",a);
  end
 endfunction
 always @*begin
  rdata=0;
  case(address)
   'h00:rdata=32'h52563233;
   'h04:rdata=configured?11:10;
   'h08:rdata=events;
   'h1c:rdata=native_status;
   'h20:rdata=native_result;
   'h30:rdata=spi_index<144?9:2;
   'h34:rdata=directory[spi_index<144?spi_index:143];
   'h48:rdata=2;
   'h4c:rdata=dma_crc;
   'h58:rdata=camera_status;
   'h5c:rdata=camera_result;
   'h64:rdata=flags;
   // Functional-test timer acceleration; no timing/performance claim is made.
   'h78:rdata=cycles<<5;
   'h84:rdata=isp_control;
   'h88:rdata=isp_peek;
   'h8c:rdata=isp_accepted;
   'h98:rdata=request_seq;
   'ha0:rdata=ui_value;
   'ha4:rdata=irq_pending;
   'hb0:rdata=cache_invalidated;
  endcase
 end
 always @(posedge clk)begin
  cycles<=cycles+1;
  if(case_id==0 && fault!=0)$fatal(1,"Unexpected firmware fault %h pc=%h",fault,cpu_pc);
  if(cycles==10000 || cycles%200000==199999)$display("PROGRESS cycles=%0d pc=%h camera=%0d dma=%0d ready=%0d",cycles,cpu_pc,camera_writes,dma_commands,ready);
  if(flags==3 && cycles%1000==0)frame_count<=frame_count+1;
  if(fire && !writing && address=='h34)spi_index<=spi_index+1;
  if(fire && writing)case(address)
   'h0c:events<=events&~wdata;
   'h10:native_address<=wdata;
   'h14:native_data<=wdata;
   'h18:begin
    native_commands<=native_commands+1;native_status<=case_id==4?4:2;
    if(wdata==1)native_result<=native_read(native_address);
    else if(wdata==2)begin
     if(native_address==32'hf9000f00)host_command<=native_data;
     if(native_address==32'hf8100008)native_control<=native_data;
     if(native_address==32'hf9000f28)begin
      if(native_data==32'h49400001)begin end
      else if(native_data[31:16]==16'h4980)isp_expected[15:0]<=native_data[15:0];
      else if(native_data[31:16]==16'h49c0)begin
       isp_expected[31:16]<=native_data[15:0];isp_accepted<={native_data[15:0],isp_expected[15:0]};isp_control<=32'h1700;
       for(integer k=0;k<768;k=k+1)if(staging[k]!==isp[k])$fatal(1,"ISP staging mismatch at %0d",k);
      end else if(native_data[31:24]==8'h49)begin
       index=(native_data>>8)&1023;if(index>=768)$fatal(1,"ISP index");staging[index]<=native_data[7:0];
      end else if(native_data[31:24]==8'h4a)begin
       index=native_data[7:0];isp_peek<={index[7:0],isp[512+index],isp[256+index],isp[index]};
      end else $fatal(1,"ISP command %h",native_data);
     end
    end else $fatal(1,"Native opcode");
   end
   'h24:spi_address<=wdata;
   'h28:spi_length<=wdata;
   'h2c:begin if(spi_address!=32'h340000 || spi_length!=576)$fatal(1,"Catalog bounds");spi_index<=0;end
   'h38:dma_source<=wdata;
   'h3c:dma_target<=wdata;
   'h40:dma_length<=wdata;
   'h44:begin
    irq_pending<=1;
    if(!configured)$fatal(1,"Data used before camera configured");
    dma_commands<=dma_commands+1;
    if(wdata<1 || wdata>4)$fatal(1,"DMA opcode");
    temp=memory_crc(dma_source,dma_length);dma_crc<=case_id==3?~temp:temp;
    if(wdata==2 && dma_target==32'he1000000)active_compute_crc<=temp;
    if(wdata==2 && dma_target==32'he1200000)active_coeff_crc<=temp;
    if(wdata==4 && (dma_length!=20480 || dma_target<1 || dma_target>65535 ||
       (style==7 ? dma_source!=32'he1200000 : dma_source!=32'he18c0000+(style[2:0]<<20))))$fatal(1,"Resident/manual coefficient command contract");
    if(wdata==2)$fatal(1,"Resident hot selection must not recopy active program/coefficients");
   end
   'h50:camera_word<=wdata;
   'h54:begin
    camera_status<=2;
    if(camera_word[15:8]!=8'h60)$fatal(1,"Unexpected camera address");
    if(wdata==2)camera_result<=case_id==1 && camera_word[31:16]==16'h3107 ? 0 : camera_registers[camera_word[31:16]];
    else if(wdata==1)begin
     if(camera_writes>=164 || camera_word[31:16]!==camera_table[camera_writes][23:8] || camera_word[7:0]!==camera_table[camera_writes][7:0])$fatal(1,"Camera table/order mismatch %0d",camera_writes);
     camera_registers[camera_word[31:16]]<=camera_word[7:0];camera_writes<=camera_writes+1;
    end else $fatal(1,"Camera opcode");
   end
   'h60:begin if(camera_writes!=164)$fatal(1,"Camera prematurely published");configured<=wdata[0];end
   'h64:flags<=wdata;
   'h68:run_state<=wdata;
   'h6c:style<=wdata;
   'h70:fault<=wdata;
   'h7c:view<=wdata;
   'h90:ready<=wdata;
   'h9c:completed_seq<=wdata;
   'ha0:begin ui_value<=wdata;ui_updates<=ui_updates+1;end
   'ha8:if(wdata[0])irq_pending<=0;
   'hac:irq_handled<=wdata;
   'hb0:if(wdata==1)cache_invalidated<=0;
   'hb8:frame_budget<=wdata;
  endcase
  if(cycles>20000000)$fatal(1,"CPU test timeout pc=%h flags=%h ready=%h fault=%h",cpu_pc,flags,ready,fault);
 end
 task automatic event_request(input[31:0]ev);
  begin
   @(negedge clk);request_seq=request_seq+1;events=ev;
   wait(completed_seq==request_seq);
   @(negedge clk);
  end
 endtask
 initial begin
  if(!$value$plusargs("CASE=%d",case_id))case_id=0;
  $readmemh("catalog.hex",directory);$readmemh("camera.hex",camera_table);$readmemh("isp.hex",isp);
  for(integer i=0;i<65536;i=i+1)camera_registers[i]=0;
  camera_registers['h3107]=8'hcd;camera_registers['h3108]=8'h6b;
  // These three fields are verified, but are not written by the frozen 222
  // table. Model the values observed/required by its successful board boot.
  camera_registers['h3018]=8'h60;camera_registers['h3031]=8'h0a;camera_registers['h3037]=8'h20;
  if(case_id==2)directory[5]=~directory[5];
  repeat(10)@(posedge clk);@(negedge clk);reset=0;
  if(case_id!=0)begin
   wait(fault!=0);
   if((case_id==1 && fault!=8'hc4) || (case_id==2 && fault!=8'h12) || (case_id==3 && fault!=8'h23) || (case_id==4 && fault!=8'hd1))$fatal(1,"Wrong failure code case %0d got %h",case_id,fault);
   if(flags!=0 || ready!=0)$fatal(1,"Fault did not revoke run permission");
   repeat(100)@(posedge clk);
   $display("PASS RV32_FAULT case=%0d code=%h cycles=%0d",case_id,fault,cycles);$finish;
  end
  wait(ready==1);repeat(20000)@(posedge clk);if(fault)$fatal(1,"Startup fault %h",fault);
  if(camera_writes!=164 || dma_commands!=29)$fatal(1,"Missing C startup responsibilities camera=%0d dma=%0d",camera_writes,dma_commands);
  $display("RV32_STARTUP cycles=%0d native=%0d dma=%0d",cycles,native_commands,dma_commands);
  event_request(1);if(flags!=3)$fatal(1,"Start not issued by C");
  event_request(4);if(view!=0 || native_control!=1)$fatal(1,"Full/half first toggle");
  event_request(4);if(view!=1 || native_control!=3)$fatal(1,"Full/half reverse toggle");
  for(integer mode_index=1;mode_index<=12;mode_index=mode_index+1)begin
   event_request(2);
   if(style[2:0]!=(mode_index%6) || flags!=3)$fatal(1,"Six-mode cycle %0d got %0d",mode_index,style[2:0]);
  end
  event_request(1);if(flags!=5)$fatal(1,"Pause");
  event_request(1);if(flags!=3)$fatal(1,"Resume");
  event_request(1);if(flags!=5)$fatal(1,"Stop before manual upload");
  @(negedge clk);cache_invalidated=1;active_compute_crc=directory[20];active_coeff_crc=directory[23];
  wait(ready==0);
  @(negedge clk);host_command=32'h4c4f4144;
  wait(dma_commands==44 && ready==1);
  if(style!=7 || flags!=5 || cache_invalidated || frame_budget!=1023)$fatal(1,"Legacy LOAD failed");
  @(negedge clk);host_command=32'h434e4e31;
  wait(dma_commands==47 && ready==1);
  if(style!=7 || flags!=5 || frame_budget!=3)$fatal(1,"Legacy CNN1 preparation");
  @(negedge clk);host_command=32'h4c495645;
  wait(flags==3);before_frame=frame_count;
  wait(flags==5 && frame_budget==1023);
  if(frame_count-before_frame<3 || frame_budget!=1023)$fatal(1,"Finite run did not stop");
  @(negedge clk);host_command=32'h4c495645;
  wait(flags==3);
  @(negedge clk);host_command=32'h53544f50;
  wait(flags==5);
  $display("RV32_LEGACY_LOAD_CNN1_LIVE_STOP_PASS DMA=%0d",dma_commands);
  if(irq_handled==0)$fatal(1,"Official PLIC IRQ never handled");
  $display("OFFICIAL_PLIC_IRQ_HANDLED count=%0d",irq_handled);

  $display("PASS RV32_C_CAMERA_WEIGHTS_SCHEDULING cycles=%0d camera=%0d dma=%0d seq=%0d",cycles,camera_writes,dma_commands,completed_seq);$finish;
 end
endmodule
