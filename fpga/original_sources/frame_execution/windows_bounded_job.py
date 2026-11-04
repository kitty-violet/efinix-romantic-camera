"""Run one offline command tree with Windows-enforced resource limits.

Only the new job is affected. No service, driver, power or global setting changes.
The child waits for assignment before it may launch the requested command.
"""
import argparse,ctypes,json,os,subprocess,sys,time,traceback
from ctypes import wintypes as W
from pathlib import Path

class Basic(ctypes.Structure):
    _fields_=[('user',ctypes.c_int64),('job_user',ctypes.c_int64),('flags',W.DWORD),
              ('min_ws',ctypes.c_size_t),('max_ws',ctypes.c_size_t),('active',W.DWORD),
              ('affinity',ctypes.c_size_t),('priority',W.DWORD),('scheduling',W.DWORD)]
class IO(ctypes.Structure):
    _fields_=[(n,ctypes.c_uint64) for n in ('read_ops','write_ops','other_ops','read_bytes','write_bytes','other_bytes')]
class Extended(ctypes.Structure):
    _fields_=[('basic',Basic),('io',IO),('process_memory',ctypes.c_size_t),
              ('job_memory',ctypes.c_size_t),('peak_process',ctypes.c_size_t),('peak_job',ctypes.c_size_t)]
class Cpu(ctypes.Structure):
    _fields_=[('flags',W.DWORD),('rate',W.DWORD)]
class Memory(ctypes.Structure):
    _fields_=[('length',W.DWORD),('load',W.DWORD)]+[(n,ctypes.c_uint64) for n in
        ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]

def kernel():
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    for name,args,rest in (
        ('CreateJobObjectW',[ctypes.c_void_p,W.LPCWSTR],W.HANDLE),
        ('CreateMutexW',[ctypes.c_void_p,W.BOOL,W.LPCWSTR],W.HANDLE),
        ('SetInformationJobObject',[W.HANDLE,ctypes.c_int,ctypes.c_void_p,W.DWORD],W.BOOL),
        ('QueryInformationJobObject',[W.HANDLE,ctypes.c_int,ctypes.c_void_p,W.DWORD,ctypes.c_void_p],W.BOOL),
        ('AssignProcessToJobObject',[W.HANDLE,W.HANDLE],W.BOOL),
        ('TerminateJobObject',[W.HANDLE,W.UINT],W.BOOL),
        ('GlobalMemoryStatusEx',[ctypes.POINTER(Memory)],W.BOOL),
        ('GetCurrentProcess',[],W.HANDLE),
        ('GetProcessAffinityMask',[W.HANDLE,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)],W.BOOL),
        ('CloseHandle',[W.HANDLE],W.BOOL),('ReleaseMutex',[W.HANDLE],W.BOOL)):
        f=getattr(k,name);f.argtypes=args;f.restype=rest
    return k
def checked(value):
    if not value:raise ctypes.WinError(ctypes.get_last_error())
    return value
def save(path,value):
    tmp=path.with_suffix(path.suffix+'.writing')
    with tmp.open('w',encoding='utf8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False);f.flush();os.fsync(f.fileno())
    # Windows readers/antivirus may briefly deny rename while opening JSON.
    # Retry that transient sharing conflict; keep the previous complete JSON.
    for attempt in range(40):
        try:os.replace(tmp,path);break
        except PermissionError:
            if attempt==39:raise
            time.sleep(.05)
def free_memory(k):
    m=Memory();m.length=ctypes.sizeof(m);checked(k.GlobalMemoryStatusEx(ctypes.byref(m)))
    return m.avail_phys

def main():
    if len(sys.argv)>1 and sys.argv[1]=='--child':
        if sys.stdin.readline().strip()!='ASSIGNED':return 125
        return subprocess.call(sys.argv[3:],cwd=sys.argv[2])
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True);p.add_argument('--cwd',type=Path,required=True)
    p.add_argument('--seconds',type=float,required=True);p.add_argument('--memory-mb',type=int,default=4096)
    p.add_argument('--reserve-mb',type=int,default=6144);p.add_argument('--cores',type=int,default=4)
    p.add_argument('--cpu-percent',type=int,default=25);p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();cmd=a.command[1:] if a.command[:1]==['--'] else a.command
    assert os.name=='nt' and cmd and 0<a.seconds<=1800
    assert 64<=a.memory_mb<=4096 and 1<=a.cores<=4 and 1<=a.cpu_percent<=25 and a.reserve_mb>=6144
    root=a.root.resolve();assert root.is_relative_to(Path('E:/fpga/tmp')) and not root.exists()
    root.mkdir();k=kernel();mutex=None;job=None;proc=None;start=time.monotonic()
    result=dict(status='PREPARING',command=cmd,cwd=str(a.cwd.resolve()),hardware_access=False,
                memory_limit_mb=a.memory_mb,reserve_mb=a.reserve_mb,cores=a.cores,
                cpu_percent=a.cpu_percent,priority='BELOW_NORMAL',timeout_seconds=a.seconds)
    try:
        mutex=checked(k.CreateMutexW(None,True,'Local\\FPGA_ArchitectureReview_BoundedOfflineJob'))
        if ctypes.get_last_error()==183:raise RuntimeError('Another bounded offline job owns the single-run lock')
        assert free_memory(k)>=a.reserve_mb*1024**2,'Host free memory below reserve before launch'
        job=checked(k.CreateJobObjectW(None,None))
        available=ctypes.c_size_t();system=ctypes.c_size_t()
        checked(k.GetProcessAffinityMask(k.GetCurrentProcess(),ctypes.byref(available),ctypes.byref(system)))
        mask=sum(1<<i for i in [i for i in range(ctypes.sizeof(ctypes.c_size_t)*8) if available.value>>i&1][:a.cores])
        assert mask
        limits=Extended();limits.basic.flags=0x2000|0x200|0x20|0x10
        limits.basic.priority=0x4000;limits.basic.affinity=mask;limits.job_memory=a.memory_mb*1024**2
        checked(k.SetInformationJobObject(job,9,ctypes.byref(limits),ctypes.sizeof(limits)))
        cpu=Cpu(5,a.cpu_percent*100)
        checked(k.SetInformationJobObject(job,15,ctypes.byref(cpu),ctypes.sizeof(cpu)))
        actual=Extended();actual_cpu=Cpu()
        checked(k.QueryInformationJobObject(job,9,ctypes.byref(actual),ctypes.sizeof(actual),None))
        checked(k.QueryInformationJobObject(job,15,ctypes.byref(actual_cpu),ctypes.sizeof(actual_cpu),None))
        assert actual.job_memory==limits.job_memory and actual.basic.affinity==mask
        assert actual_cpu.flags==5 and actual_cpu.rate==cpu.rate
        result.update(affinity_mask=mask,limits_readback_verified=True)
        with (root/'command.log').open('x',encoding='utf8') as log:
            proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--child',str(a.cwd.resolve()),*cmd],
                stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,text=True,
                creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
            checked(k.AssignProcessToJobObject(job,int(proc._handle)))
            result.update(status='RUNNING',pid=proc.pid);save(root/'progress.json',result)
            proc.stdin.write('ASSIGNED\n');proc.stdin.flush();proc.stdin.close()
            reason=None
            while proc.poll() is None:
                free=free_memory(k);elapsed=time.monotonic()-start
                checked(k.QueryInformationJobObject(job,9,ctypes.byref(actual),ctypes.sizeof(actual),None))
                result.update(seconds=round(elapsed,3),free_ram_mb=round(free/1024**2,1),peak_job_mb=round(actual.peak_job/1024**2,1))
                try:save(root/'progress.json',result)
                except PermissionError:
                    result['progress_write_conflicts']=result.get('progress_write_conflicts',0)+1
                if free<a.reserve_mb*1024**2:reason='HOST_MEMORY_RESERVE_STOP'
                elif elapsed>a.seconds:reason='HOST_TIMEOUT_STOP'
                if reason:checked(k.TerminateJobObject(job,124));break
                time.sleep(1)
            code=proc.wait(timeout=10)
            checked(k.QueryInformationJobObject(job,9,ctypes.byref(actual),ctypes.sizeof(actual),None))
            result.update(status=reason or ('COMMAND_PASS' if code==0 else 'COMMAND_FAILED'),exit_code=code,
                seconds=round(time.monotonic()-start,3),peak_job_mb=round(actual.peak_job/1024**2,1))
        returncode=0 if result['status']=='COMMAND_PASS' else 1
    except Exception as exc:
        result.update(status='GUARD_FAILURE_NO_UNBOUNDED_FALLBACK',error=repr(exc),traceback=traceback.format_exc());returncode=1
    finally:
        if job:k.TerminateJobObject(job,125);k.CloseHandle(job)
        if proc and proc.poll() is None:proc.kill();proc.wait(timeout=10)
        if mutex:k.ReleaseMutex(mutex);k.CloseHandle(mutex)
        result['seconds']=round(time.monotonic()-start,3);save(root/'result.json',result)
    print(json.dumps(result,indent=2,ensure_ascii=False));return returncode
if __name__=='__main__':raise SystemExit(main())
