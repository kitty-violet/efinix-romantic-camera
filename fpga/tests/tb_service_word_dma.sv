`timescale 1ns/1ps
module tb_service_word_dma;
 reg clk=0,reset=1,start=0;always #3.333 clk=~clk;
 reg[2:0]op=0;reg[31:0]src=0,dst=0,bytes=0;
 wire sr,busy,done,error,poisoned;wire[7:0]fault;wire[31:0]crc;
 wire cv,cw;wire[31:0]ca,cd;reg cr=0,rv=0,re=0;reg[31:0]rd=0;
 wire ss,spr;wire[23:0]spa,spl;
 reg spv=0,spe=0;reg[31:0]spd=0;
 service_word_dma #(.BUS_TIMEOUT(128),.DIAG_POLL_LIMIT(16))dut(
 .clk(clk),.reset(reset),.start(start),.operation(op),.source(src),.target(dst),.byte_count(bytes),
 .start_ready(sr),.busy(busy),.done(done),.error(error),.poisoned(poisoned),.fault(fault),.crc_result(crc),.word_result(),
 .cmd_valid(cv),.cmd_write(cw),.cmd_address(ca),.cmd_data(cd),.cmd_ready(cr),
 .rsp_valid(rv),.rsp_error(re),.rsp_data(rd),.spi_start(ss),.spi_start_ready(1'b1),
 .spi_address(spa),.spi_length(spl),.spi_valid(spv),.spi_error(spe),.spi_data(spd),.spi_ready(spr));
 reg[31:0]memory[0:8191];reg[63:0]weights[0:2559];
 reg[31:0]rng=32'h9174ce20,pa=0,pd=0;reg pw=0,pending=0;
 integer countdown=0,cycles=0,transactions=0,inject_error=-1;
 integer load_index=0,diag_index=0,polls=0,spi_index=0,spi_words=0;
 reg[31:0]load_lo=0,load_hi=0,diag_address=0;
 reg[64:0]held;
 reg stalled=0;
 function automatic[31:0]crc32(input integer first,input integer length);
  reg[31:0]c,w;
  begin c=32'hffffffff;
   for(integer n=0;n<length;n=n+1)begin
    w=memory[first+n];c=c^w;
    for(integer b=0;b<32;b=b+1)c=(c>>1)^(c[0]?32'hedb88320:0);
   end
   crc32=~c;
  end
 endfunction
 always @(posedge clk)begin
  cycles<=cycles+1;
  rng<={rng[30:0],rng[31]^rng[21]^rng[1]^rng[0]};
  cr<=!pending && rng[0];rv<=0;re<=0;
  if(stalled && cv && {cw,ca,cd}!==held)$fatal(1,"Unstable native offer under backpressure");
  stalled<=cv && !cr;held<={cw,ca,cd};
  if(cv && cr)begin
   if(pending)$fatal(1,"Multiple outstanding transactions");
   pending<=1;countdown<=rng[5:2]+1;pa<=ca;pd<=cd;pw<=cw;transactions<=transactions+1;
  end
  if(pending)begin
   cr<=0;
   if(countdown!=0)countdown<=countdown-1;
   else begin
    pending<=0;rv<=1;re<=transactions==inject_error;rd<=0;
    if(pa[31:24]==8'he1)begin
     if(pw)memory[pa[14:2]]<=pd;else rd<=memory[pa[14:2]];
    end else if(pa[31:16]==16'hf810)begin
     if(pw)case(pa[7:0])
      'h24:load_index=pd;
      'h28:load_lo=pd;
      'h2c:load_hi=pd;
      'h30:begin
       if(load_index<0 || load_index>2559 || pd!=(load_index==2559?3:1))$fatal(1,"Coefficient push/index/last");
       weights[load_index]={load_hi,load_lo};
      end
      'h34:begin diag_address=pd;diag_index=((pd>>10)&7)+(pd&511)*5;end
      'h38:polls=0;
      default:$fatal(1,"Unexpected APB write %h",pa);
     endcase
     else case(pa[7:0])
      'h04:rd<=1;
      'h3c:begin rd<=polls>1;polls=polls+1;end
      'h4c:rd<=diag_address;
      'h68:rd<=7;
      'h40:rd<=weights[diag_index][31:0];
      'h44:rd<=weights[diag_index][63:32];
      default:$fatal(1,"Unexpected APB read %h",pa);
     endcase
    end else $fatal(1,"Address escape %h",pa);
   end
  end
  if(ss)begin spi_index<=0;spi_words<=spl>>2;spv<=0;end
  else if(spi_index<spi_words)begin
   if(spv && spr)begin spv<=0;spi_index<=spi_index+1;end
   else if(!spv && rng[2])begin spv<=1;spd<=memory[spi_index];end
  end
  if(cycles>3000000)$fatal(1,"DMA timeout state=%0d index=%0d",dut.state,dut.index);
 end
 task automatic launch(input[2:0]operation,input[31:0]source,target,len);
  begin
   wait(sr);@(negedge clk);op=operation;src=source;dst=target;bytes=len;start=1;
   @(negedge clk);start=0;wait(done);@(negedge clk);
  end
 endtask
 initial begin
  for(integer i=0;i<8192;i=i+1)memory[i]=i*32'h71a49e2d^32'h6829e6ad;
  repeat(5)@(negedge clk);reset=0;
  launch(2,32'he1800000,32'he1804000,256);
  if(error || crc!==crc32(0,64))$fatal(1,"DDR copy CRC");
  for(integer i=0;i<64;i=i+1)if(memory[4096+i]!==memory[i])$fatal(1,"DDR copy payload");
  launch(3,32'he1804000,0,256);if(error || crc!==crc32(4096,64))$fatal(1,"Readback CRC");
  launch(1,32'h340000,32'he1804000,128);if(error || crc!==crc32(0,32))$fatal(1,"SPI copy CRC");
  launch(4,32'he1200000,7,20480);
  if(error || crc!==crc32(0,5120))$fatal(1,"Coefficient CRC/readback fault=%h",fault);
  for(integer i=0;i<2560;i=i+1)if(weights[i]!=={memory[i*2+1],memory[i*2]})$fatal(1,"Coefficient payload %0d",i);
  launch(4,32'he1dc0000,7,20480);
  if(error || crc!==crc32(0,5120))$fatal(1,"Resident slot5 coefficient CRC/readback");
  $display("PASS DMA_COPY_CRC_COEFFICIENTS transactions=%0d cycles=%0d",transactions,cycles);
  launch(2,32'he1800000,32'he1804000,3);
  if(!error || fault!=1)$fatal(1,"Misaligned transfer accepted");
  repeat(5)@(negedge clk);reset=1;repeat(5)@(negedge clk);reset=0;
  inject_error=transactions+2;
  launch(2,32'he1800000,32'he1804000,256);
  if(!error || fault!=8'hd1)$fatal(1,"Bus response error not retained");
  repeat(50)@(posedge clk);if(cv || busy)$fatal(1,"DMA continued after error");
  $display("PASS DMA_BOUNDS_FAULT_DRAIN");$finish;
 end
endmodule
