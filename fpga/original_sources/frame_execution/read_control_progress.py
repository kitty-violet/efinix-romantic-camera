"""OFP2 extended first fault + current control progress, selectors 2/3 only."""
import time
import read_first_fault as old
from read_control_snapshot import SnapshotPending

MAGIC=0x4f465032
SCAN_BITS=386

def decode(raw):
    if not isinstance(raw,int) or isinstance(raw,bool) or raw<0 or raw>=1<<SCAN_BITS:
        raise ValueError('Invalid OFP2 scan length/value')
    value=raw>>2
    if value&0xffffffff!=MAGIC or value>>40&255!=2:
        raise ValueError('OFP2 signature/version mismatch')
    # First-fault layout is identical; explicitly adapt only the checked header.
    compatible=(value&((1<<256)-1))&~(0xffffffff|(255<<40))
    compatible|=old.MAGIC|(1<<40)
    result=old.decode(compatible<<2)
    result.update(raw_hex=f'{value:096x}',format='OFP2',live=None)
    if result['valid']:
        flags=value>>256&0xffffffff; state=value>>288&0xffffffff
        counts=value>>320&0xffffffff
        if state>>29:raise ValueError('Reserved live state bits set')
        result['live']=dict(flags=old.flags(flags),control_state=old.state(state&1023),
                            boot_state=state>>10&63,bus_age=state>>16&8191,
                            received_bits=counts&255,received_last=counts>>8&255,
                            accepted_commands=counts>>16&255,replies=counts>>24&255,
                            ticks=value>>352&0xffffffff)
    return result

def fresh(transport,history,save,seconds=1):
    deadline=time.monotonic()+seconds
    def scan(selector):
        row=decode(transport.scan(selector,SCAN_BITS));history.append(dict(selector=selector,**row));save();return row
    previous=scan(3)
    for _ in range(8):
        if not previous['busy'] or time.monotonic()>=deadline:break
        previous=scan(3)
    if previous['busy']:raise SnapshotPending('Previous snapshot remains pending')
    scan(2)
    for _ in range(8):
        if time.monotonic()>=deadline:break
        current=scan(3)
        if current['valid'] and not current['busy'] and current['sequence']==(previous['sequence']+1)&255:return current
    raise SnapshotPending('No fresh OFP2 record; prefix retained')

if __name__=='__main__':
    # Keep the established finite worker/transport and no-bus-access contract.
    old.fresh=fresh
    # The worker must re-enter this decoder, not the legacy 258-bit script.
    old.__file__=__file__
    old.main()
