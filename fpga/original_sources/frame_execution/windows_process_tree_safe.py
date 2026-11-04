"""Bounded stop of owned Windows processes, rejecting reused parent PIDs."""
import ctypes
from ctypes import wintypes
from windows_process_tree import process_tree

def descendants(rows,root_pid,root_created):
    byid={r['pid']:r for r in rows}
    selected={root_pid:root_created}; pending=[root_pid]
    for parent in pending:
        for row in rows:
            if row['pid'] not in selected and row['ppid']==parent and row['created']>=selected[parent]:
                selected[row['pid']]=row['created'];pending.append(row['pid'])
    return [byid[p] for p in reversed(pending) if p!=root_pid]

def stop_launched_tree(proc):
    if proc.poll() is not None:return dict(already_exited=True,processes=[])
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];api.OpenProcess.restype=wintypes.HANDLE
    api.GetProcessTimes.argtypes=[wintypes.HANDLE,*([ctypes.POINTER(wintypes.FILETIME)]*4)]
    api.GetProcessId.argtypes=[wintypes.HANDLE];api.GetProcessId.restype=wintypes.DWORD
    api.TerminateProcess.argtypes=[wintypes.HANDLE,wintypes.UINT]
    api.CloseHandle.argtypes=[wintypes.HANDLE]
    def created(handle):
        times=[wintypes.FILETIME() for _ in range(4)]
        if not api.GetProcessTimes(handle,*[ctypes.byref(t) for t in times]):raise ctypes.WinError(ctypes.get_last_error())
        return (times[0].dwHighDateTime<<32)|times[0].dwLowDateTime
    root_handle=wintypes.HANDLE(int(proc._handle))
    assert api.GetProcessId(root_handle)==proc.pid
    root_created=created(root_handle)
    # The legacy enumeration is a superset only. No returned PID is trusted.
    candidates=process_tree(proc.pid);handles={};rows=[];stopped=[];rejected=[]
    class Entry(ctypes.Structure):
        _fields_=[('dwSize',wintypes.DWORD),('cntUsage',wintypes.DWORD),('th32ProcessID',wintypes.DWORD),('th32DefaultHeapID',ctypes.c_size_t),('th32ModuleID',wintypes.DWORD),('cntThreads',wintypes.DWORD),('th32ParentProcessID',wintypes.DWORD),('pcPriClassBase',wintypes.LONG),('dwFlags',wintypes.DWORD),('szExeFile',wintypes.WCHAR*260)]
    api.CreateToolhelp32Snapshot.argtypes=[wintypes.DWORD,wintypes.DWORD];api.CreateToolhelp32Snapshot.restype=wintypes.HANDLE
    api.Process32FirstW.argtypes=[wintypes.HANDLE,ctypes.POINTER(Entry)];api.Process32NextW.argtypes=[wintypes.HANDLE,ctypes.POINTER(Entry)]
    snapshot=api.CreateToolhelp32Snapshot(2,0)
    if snapshot==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    parents={};entry=Entry();entry.dwSize=ctypes.sizeof(entry)
    try:
        more=api.Process32FirstW(snapshot,ctypes.byref(entry))
        while more:
            parents[entry.th32ProcessID]=entry.th32ParentProcessID
            more=api.Process32NextW(snapshot,ctypes.byref(entry))
    finally:api.CloseHandle(snapshot)
    try:
        for row in candidates:
            pid=row['pid']
            if pid==proc.pid:continue
            handle=api.OpenProcess(0x1001,False,pid)
            if not handle:
                rejected.append(dict(pid=pid,reason='exited or inaccessible; not killed'));continue
            handles[pid]=handle
            rows.append(dict(row,ppid=parents.get(pid,-1),created=created(handle)))
        verified=descendants(rows,proc.pid,root_created)
        selected={r['pid'] for r in verified}
        rejected.extend(dict(r,reason='not in creation-time-validated ownership chain') for r in rows if r['pid'] not in selected)
        for row in verified:
            handle=handles[row['pid']]
            assert api.GetProcessId(handle)==row['pid'] and created(handle)==row['created']
            if api.TerminateProcess(handle,1):stopped.append(row)
            else:rejected.append(dict(row,reason='process already exited or termination denied'))
        if proc.poll() is None:proc.terminate()
        proc.wait(timeout=20)
        return dict(status='OWNED_HANDLES_AND_CREATION_TIMES_CHECKED',root_pid=proc.pid,root_created=root_created,processes=stopped,rejected=rejected,exit_code=proc.returncode)
    finally:
        for handle in handles.values():api.CloseHandle(handle)
