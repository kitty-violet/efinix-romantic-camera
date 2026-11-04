"""Focused OFP3 gates; reuse existing four-state and exact-map test machinery."""
import argparse,json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_ddr_combined import audit_delta,once,COUNTERS

def driver_run(source,code,root,args):
    driver=root.parent/(root.name+'-combined-driver.py')
    assert not driver.exists() and not root.exists()
    compile(code,str(driver),'exec');driver.write_text(code,encoding='utf-8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
        driver_sha256=sha(driver),wrapper_sha256=sha(__file__)),indent=2))
    sys.argv=[str(driver),*map(str,args)]
    try:runpy.run_path(str(driver),run_name='__main__')
    except SystemExit as e:
        if e.code not in (None,0):raise

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('rtl','startup','serial'))
    p.add_argument('--board',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--startup',type=Path);a=p.parse_args()
    board=a.board.resolve();root=a.root.resolve();audit_delta(board)
    assert root.is_relative_to(ROOT/'tmp/frame-execution')
    if a.mode=='serial':assert root.parent==ROOT/'tmp/frame-execution'
    if a.mode=='rtl':
        source=HERE/'validate_control_progress.py';code=source.read_text()
        code=once(code,'from control_command_repair import audit_delta','from control_ddr_combined import audit_delta')
        code=once(code,'from read_control_progress import decode','from read_control_ddr_progress import decode')
        for old,new in (('[319:0]','[383:0]'),('[385:0]','[449:0]'),('386','450'),("32'h4f465032","32'h4f465033"),
                        ('raw[49:42]!==2','raw[49:42]!==3'),('raw[385:66]','raw[449:66]')):
            assert old in code;code=code.replace(old,new)
        code=once(code,'wire [383:0] payload={ticks,counts,state,flags,first};',
            "reg [31:0] diagnostic_board_status=0;reg [1:0] diagnostic_shift_events=0;\n"+COUNTERS+
            'wire [383:0] payload={ddr_lock_drops,system_reset_events,shift_forwarded,shift_requests,diagnostic_board_status,ticks,counts,state,flags,first};')
        code=once(code,' identity;snapshot;',r''' identity;
 diagnostic_board_status=32'h001daaed;diagnostic_shift_events=1;repeat(10)@(negedge clk);
 diagnostic_shift_events=2;repeat(10)@(negedge clk);
 diagnostic_shift_events=0;repeat(10)@(negedge clk);
 diagnostic_board_status=32'h001daae4;repeat(10)@(negedge clk);
 if({ddr_lock_drops,system_reset_events,shift_forwarded,shift_requests}!==32'h01010101)$fatal(1,"DDR event count mismatch");
 // A held enable must count once; snapshot scans must not create events.
 snapshot;
 if({ddr_lock_drops,system_reset_events,shift_forwarded,shift_requests}!==32'h01010101)$fatal(1,"idle event growth");''')
        code=once(code,' if(commands!=2)$fatal',r''' // Modulo wrap: 255 more complete sampled request/forward pulses.
 repeat(255)begin
  diagnostic_shift_events=3;repeat(5)@(negedge clk);
  diagnostic_shift_events=0;repeat(5)@(negedge clk);
 end
 if({shift_forwarded,shift_requests}!==16'h0000 || system_reset_events!==1 || ddr_lock_drops!==1)$fatal(1,"DDR event wrap/reset retention");
 if(commands!=2)$fatal''')
        code=once(code,"assert len(records)==2 and", "assert records[0]['ddr']['raw_done'] and records[0]['ddr']['raw_pass']\n assert not records[0]['ddr']['system_reset_n'] and not records[0]['ddr']['ddr_pll_locked']\n assert records[0]['ddr']['failure_log']==0xaa and records[0]['ddr']['sampled_shift_requests']==1\n assert records[1]['ddr']['sampled_shift_forwarded']==1\n assert len(records)==2 and")
        code=code.replace("status='COMMAND_PROGRESS_RTL_SERIAL_RESET_PASS'","status='CONTROL_DDR_RTL_SERIAL_EVENTS_RESET_PASS'")
        driver_run(source,code,root,['--board',board,'--root',root])
    elif a.mode=='startup':
        source=HERE/'validate_command_progress_mapped.py';code=source.read_text()
        code=once(code,'from control_command_repair import audit_delta','from control_ddr_combined import audit_delta')
        # A larger independent observation bus retains more real cells.
        code=code.replace('len(selected)<5500','len(selected)<6000')
        driver_run(source,code,root,['--board',board,'--root',root])
    else:
        startup=a.startup.resolve();source=ROOT/'tmp/board-test/check-mapped-jtag-regression.py';code=source.read_text()
        for old,new in (('from read_first_fault import decode','from read_control_ddr_progress import decode'),
                        ('[257:0]','[449:0]'),('258','450'),("32'h4f465331","32'h4f465033"),
                        ('raw[35:34]!==1','raw[35:34]!==1 || raw[49:42]!==3')):
            assert old in code;code=code.replace(old,new)
        before="sources={'old':R/'tmp/frame-execution/first-fault-reference-physical1/startup','new':R/'tmp/frame-execution/shift-guard-physical2/startup'}"
        code=once(code,before,"sources={'new':Path("+repr(str(startup))+")}")
        code=once(code,' reg found;',' reg found;reg [449:0] first_reply;')
        code=once(code," scan(1,36,raw);\n if(raw[2]===1'b1)begin",
            " scan(1,36,first_reply);scan(1,36,raw);\n if(first_reply[2]===1'b1 && first_reply[35:2]===raw[35:2])begin")
        code=once(code,'task snap(input integer expected_capture);',r'''
task check_quiet;
 reg [7:0] accepts,replies;
 begin
  force dut.\cpu/boot_valid =1'b0;
  repeat(20)@(negedge CLOCK);
  accepts=dut.\cpu/observation_accepts ;replies=dut.\cpu/observation_replies ;
  repeat(4096)begin
   @(negedge CLOCK);
   if(dut.\cpu/state_oh !==10'h001 || dut.\cpu/remote_rsp !==0 ||
      dut.\cpu/observation_accepts !==accepts || dut.\cpu/observation_replies !==replies)
    $fatal(1,"idle APB/reply repetition state=%h",dut.\cpu/state_oh );
  end
  release dut.\cpu/boot_valid ;
 end
endtask
task snap(input integer expected_capture);''')
        marker="readword(IDENTITY,32'h54493630,1);\n snap(0);"
        assert code.count(marker)==2
        code=code.replace(marker,"readword(IDENTITY,32'h54493630,1);\n check_quiet; snap(0);")
        code=once(code,"readword(MAIL,32'h00000001,1);","readword(MAIL,32'h00000001,1);check_quiet;")
        driver_run(source,code,root,[root.name,'new'])
        result=json.loads((root/'new/result.json').read_text());records=result['snapshots']
        assert [r['live']['accepted_commands'] for r in records]==[1,2,4]
        assert [r['live']['replies'] for r in records]==[1,2,4]
        (root/'progress_counter_checks.json').write_text(json.dumps(dict(status='CONTROL_DDR_MAPPED_SERIAL_IDLE_RESET_PASS',
            accepted=[1,2,4],replies=[1,2,4],quiet_checks=3,quiet_cycles_each=4096,
            source_result_sha256=sha(root/'new/result.json'),source_map_sha256=sha(board/'outflow/ti60f225_oob.map.v'),hardware_access=False),indent=2))
        (root/'paired_reply_result.json').write_text(json.dumps(dict(status='MAPPED_PAIRED_REPLY_STARTUP_FAULT_RESET_PASS',
            source_map_sha256=sha(board/'outflow/ti60f225_oob.map.v'),result_sha256=sha(root/'new/result.json'),hardware_access=False),indent=2))
if __name__=='__main__':main()
