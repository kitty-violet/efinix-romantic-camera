"""Stop only a launched wrapper and its recorded Windows descendants."""
import ctypes,subprocess
from ctypes import wintypes

def process_tree(root_pid):
    class Entry(ctypes.Structure):
        _fields_=[('dwSize',wintypes.DWORD),('cntUsage',wintypes.DWORD),
                  ('th32ProcessID',wintypes.DWORD),('th32DefaultHeapID',ctypes.c_size_t),
                  ('th32ModuleID',wintypes.DWORD),('cntThreads',wintypes.DWORD),
                  ('th32ParentProcessID',wintypes.DWORD),('pcPriClassBase',wintypes.LONG),
                  ('dwFlags',wintypes.DWORD),('szExeFile',wintypes.WCHAR*260)]
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.CreateToolhelp32Snapshot.argtypes=[wintypes.DWORD,wintypes.DWORD]
    api.CreateToolhelp32Snapshot.restype=wintypes.HANDLE
    api.Process32FirstW.argtypes=[wintypes.HANDLE,ctypes.POINTER(Entry)]
    api.Process32NextW.argtypes=[wintypes.HANDLE,ctypes.POINTER(Entry)]
    api.CloseHandle.argtypes=[wintypes.HANDLE]
    snap=api.CreateToolhelp32Snapshot(2,0)
    if snap==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    parents={};entry=Entry();entry.dwSize=ctypes.sizeof(entry)
    try:
        more=api.Process32FirstW(snap,ctypes.byref(entry))
        while more:
            parents[entry.th32ProcessID]=(entry.th32ParentProcessID,entry.szExeFile)
            more=api.Process32NextW(snap,ctypes.byref(entry))
    finally:api.CloseHandle(snap)
    found=[int(root_pid)];seen=set(found)
    for parent in found:
        for pid,(ppid,name) in parents.items():
            if ppid==parent and pid not in seen:found.append(pid);seen.add(pid)
    return [dict(pid=pid,name=parents.get(pid,(None,''))[1]) for pid in reversed(found)]

def stop_launched_tree(proc):
    if proc.poll() is not None:return dict(already_exited=True,processes=[])
    # Snapshot descendants while our still-open Popen handle owns root identity.
    rows=process_tree(proc.pid)
    ids=','.join(str(row['pid']) for row in rows)
    cmd="Stop-Process -Id "+ids+" -Force -ErrorAction SilentlyContinue"
    result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',cmd],
                          capture_output=True,text=True,timeout=20)
    proc.wait(timeout=20)
    return dict(processes=rows,exit_code=result.returncode,output=result.stdout,error=result.stderr)
