"""Use four separate physical cores without raising the existing Job limits.

Topology API: https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/nf-sysinfoapi-getlogicalprocessorinformation
Only this launcher/descendants' affinity is changed; no global configuration.
"""
import ctypes,hashlib,json,os,sys
from ctypes import wintypes as W
from pathlib import Path
import windows_bounded_job as guard

class Processor(ctypes.Structure):
    _fields_=[('mask',ctypes.c_size_t),('relationship',W.DWORD),('reserved',ctypes.c_uint64*2)]

def topology():
    assert os.name=='nt' and ctypes.sizeof(ctypes.c_size_t)==8
    k=guard.kernel()
    fn=k.GetLogicalProcessorInformation
    fn.argtypes=[ctypes.c_void_p,ctypes.POINTER(W.DWORD)];fn.restype=W.BOOL
    size=W.DWORD()
    assert not fn(None,ctypes.byref(size)) and ctypes.get_last_error()==122
    buf=ctypes.create_string_buffer(size.value)
    guard.checked(fn(buf,ctypes.byref(size)))
    assert ctypes.sizeof(Processor)==32 and size.value%32==0
    rows=(Processor*(size.value//32)).from_buffer(buf)
    cores=sorted(int(r.mask) for r in rows if r.relationship==0)
    available=ctypes.c_size_t();system=ctypes.c_size_t()
    guard.checked(k.GetProcessAffinityMask(k.GetCurrentProcess(),ctypes.byref(available),ctypes.byref(system)))
    # Reject overlapping or unknown topology instead of guessing SMT numbering.
    union=0
    for mask in cores:
        assert mask and not union&mask
        union|=mask
    assert union==system.value and len(cores)>=4
    choices=[m&available.value for m in cores if m&available.value]
    assert len(choices)>=4
    selected=sum(m&-m for m in choices[:4])
    assert selected.bit_count()==4 and selected&available.value==selected
    return k,dict(core_masks=cores,available_mask=available.value,
        selected_mask=selected,logical_processors=union.bit_count(),physical_cores=len(cores),
        selected_physical_cores=4,limits_unchanged=True,hardware_access=False)

def main():
    k,record=topology()
    record['launcher_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    record['original_guard_sha256']=hashlib.sha256(Path(guard.__file__).read_bytes()).hexdigest()
    if sys.argv[1:]==['--inspect']:
        print(json.dumps(record,indent=2));return 0
    assert '--root' in sys.argv and '--cores' in sys.argv
    assert sys.argv[sys.argv.index('--cores')+1]=='4'
    root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
    assert root.is_relative_to(Path('E:/fpga/tmp')) and not root.exists()
    fn=k.SetProcessAffinityMask;fn.argtypes=[W.HANDLE,ctypes.c_size_t];fn.restype=W.BOOL
    guard.checked(fn(k.GetCurrentProcess(),record['selected_mask']))
    original=guard.save
    def save(path,data):
        if path.parent==root and path.name in ('progress.json','result.json'):
            data=dict(data,physical_core_selection=record)
        original(path,data)
    guard.save=save
    return guard.main()

if __name__=='__main__':raise SystemExit(main())
