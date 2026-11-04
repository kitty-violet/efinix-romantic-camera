"""OFP3: OFP2 control evidence plus independent public DDR status, no APB."""
import time
import read_control_progress as previous
import read_first_fault as worker
from read_control_snapshot import SnapshotPending

MAGIC=0x4f465033
SCAN_BITS=450

def decode(raw):
    if not isinstance(raw,int) or isinstance(raw,bool) or raw<0 or raw>=1<<SCAN_BITS:
        raise ValueError('Invalid OFP3 scan length/value')
    value=raw>>2
    if value&0xffffffff!=MAGIC or value>>40&255!=3:
        raise ValueError('OFP3 signature/version mismatch')
    compatible=(value&((1<<384)-1))&~(0xffffffff|(255<<40))
    result=previous.decode((compatible|previous.MAGIC|(2<<40))<<2)
    result.update(format='OFP3',raw_hex=f'{value:0112x}',ddr=None)
    # The old header bit is qualification, not the IP's raw completion signal.
    result['levels']['ddr_qualified']=result['levels'].pop('cal_done_raw')
    if result['valid']:
        status=value>>384&0xffffffff;events=value>>416&0xffffffff
        names=('system_reset_n','key1_n','mipi_pll_locked','ddr_pll_locked','user_pll_locked',
               'qualified_synchronized','raw_done','raw_pass')
        result['ddr']={n:bool(status>>i&1) for i,n in enumerate(names)}
        result['ddr'].update(raw_status=f'{status:08x}',failure_log=status>>8&255,
            calibration_shift_value=status>>16&7,dram_reset_n=bool(status>>19&1),cke=bool(status>>20&1),
            sampled_shift_requests=events&255,sampled_shift_forwarded=events>>8&255,
            sampled_system_reset_events=events>>16&255,sampled_ddr_lock_drops=events>>24&255,
            event_counter_modulus=256,
            status_semantics='Public multi-bit status sampled through two FFs; compare settled snapshots. Shift edges are not physical PLL completion acknowledgements.')
    return result

def fresh(transport,history,save,seconds=1):
    deadline=time.monotonic()+seconds
    def scan(sel):
        r=decode(transport.scan(sel,SCAN_BITS));history.append(dict(selector=sel,**r));save();return r
    before=scan(3)
    for _ in range(8):
        if not before['busy'] or time.monotonic()>=deadline:break
        before=scan(3)
    if before['busy']:raise SnapshotPending('Previous OFP3 request pending')
    scan(2)
    for _ in range(8):
        if time.monotonic()>=deadline:break
        current=scan(3)
        if current['valid'] and not current['busy'] and current['sequence']==(before['sequence']+1)&255:return current
    raise SnapshotPending('No fresh OFP3 snapshot; valid prefix retained')

if __name__=='__main__':
    worker.fresh=fresh;worker.__file__=__file__;worker.main()
