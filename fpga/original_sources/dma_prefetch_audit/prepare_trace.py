from pathlib import Path
import hashlib,json,shutil
R=Path(__file__).resolve().parent
B=Path('E:/fpga/workspaces/ble-button-control/s4r1-native')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (R/'rtl').exists()
shutil.copytree(B/'integration-rtl',R/'rtl')
shutil.copytree(B/'vectors/frame-dma-reordered4',R/'vectors')
for name in ['compile_schedule.py','compile_descriptors.py','reorder_dma.py','coefficients.py']:
 shutil.copy2(B/name,R/name)
shutil.copytree(B/'data/schedule4',R/'schedule')
tb=(B/'tests/tb_s4r1_frame.sv').read_text()
monitor='''
 integer audit_fd,audit_tick=0,audit_active=0,wait_start=-1,wait_c=-1,wait_d=-1,wait_req=-1;
 integer overlap_count[0:3],dma_buckets[0:19],compute_buckets[0:7];
 integer dep,category;
 initial begin
  audit_fd=$fopen("events.txt","w");
  for(integer z=0;z<4;z++)overlap_count[z]=0;
  for(integer z=0;z<20;z++)dma_buckets[z]=0;
  for(integer z=0;z<8;z++)compute_buckets[z]=0;
 end
 always @(posedge clk)begin
  if(dut.start_fire)audit_active=1;
  if(audit_active && !frame_valid)begin
   audit_tick=audit_tick+1;
   overlap_count[{dut.dp_busy,dut.dbusy}]=overlap_count[{dut.dp_busy,dut.dbusy}]+1;
   dep=int'(dut.cdata[13:0]);
   if(wait_start>=0 && !(dut.issue.active && dut.issue.cstate==0 && dut.cv && dep>int'(dma_completed) && wait_c==int'(compute_completed) && wait_d==int'(dma_completed)))begin
    $fdisplay(audit_fd,"WAIT %0d %0d %0d %0d %0d",wait_start,audit_tick,wait_c,wait_d,wait_req);wait_start=-1;
   end
   if(dut.issue.active && dut.issue.cstate==0 && dut.cv && dep>int'(dma_completed) && wait_start<0)begin
    wait_start=audit_tick;wait_c=int'(compute_completed);wait_d=int'(dma_completed);wait_req=dep;
   end
   if(dut.compute_begin)$fdisplay(audit_fd,"CS %0d %0d %0d %0d",audit_tick,compute_completed,dma_completed,dut.debug_program_index);
   if(dut.compute_end)$fdisplay(audit_fd,"CE %0d %0d",audit_tick,compute_completed);
   if(dut.dvalid && dut.dready)$fdisplay(audit_fd,"DS %0d %0d %0d",audit_tick,dma_completed,compute_completed);
   if(dut.dma_end)$fdisplay(audit_fd,"DE %0d %0d",audit_tick,dma_completed);
   if(dut.dbusy && !dut.dp_busy)begin
    category=0;
    case(dut.core.dma.state)
     1:category=arready ? 1:2;
     2:category=!rvalid ? 3:!rready ? 4:5;
     3:category=awready ? 6:7;
     4:category=!wvalid ? 8:!wready ? 9:10;
     5:category=bvalid ? 11:12;
     6:category=dut.core.dma.drained ? 13:14;
     7:category=15;
     default:category=16;
    endcase
    dma_buckets[category]=dma_buckets[category]+1;
   end
   if(dut.dp_busy && !dut.dbusy)begin
    category=!dut.dv ? 0:dut.ddata[13:0]>compute_completed ? 1:dut.issue.stop_new ? 2:dut.issue.dwait ? 3:4;
    compute_buckets[category]=compute_buckets[category]+1;
   end
  end
 end
 task audit_finish;
 begin
  if(production_cycles!=4357918)$fatal(1,"AUDIT_CHANGED_TRACE cycles=%0d",production_cycles);
  for(integer z=0;z<4;z++)$display("AUDIT_OVERLAP %0d %0d",z,overlap_count[z]);
  for(integer z=0;z<20;z++)$display("AUDIT_DMA %0d %0d",z,dma_buckets[z]);
  for(integer z=0;z<8;z++)$display("AUDIT_COMPUTE %0d %0d",z,compute_buckets[z]);
  $fclose(audit_fd);
 end endtask
'''
tb=tb.replace(' integer start_tick=0,checkpoint_tick=0;',monitor+'\n integer start_tick=0,checkpoint_tick=0;')
tb=tb.replace('$display("PASS S4R1_FULL_COMPUTE_FRAME','audit_finish();\n  $display("PASS S4R1_FULL_COMPUTE_FRAME')
(R/'tb_s4r1_frame.sv').write_text(tb,'utf8')
runner=(B/'test_profile.py').read_text().replace("d=R/'evidence'/('profile'+sys.argv[1])","d=R/('trace'+sys.argv[1])")
runner=runner.replace("R/'integration-rtl'","R/'rtl'").replace("R/'profile/tb_s4r1_frame.sv'","R/'tb_s4r1_frame.sv'").replace("R/'vectors/frame-pixel'","R/'vectors'")
(R/'run_trace.py').write_text(runner,'utf8')
files={str(p.relative_to(R)):sha(p)for p in R.rglob('*')if p.is_file()}
(R/'snapshot.json').write_text(json.dumps(dict(source=str(B),files=files,scope='Independent audit snapshot of verified DMA4 plus original runtime1; no production modification.'),indent=2),'utf8')
print('AUDIT_SNAPSHOT_READY',len(files))
