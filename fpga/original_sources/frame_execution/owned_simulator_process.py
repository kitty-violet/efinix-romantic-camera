"""Own one Windows Job for a spawned simulator; never enumerate or kill by name."""
import ctypes
import os
import subprocess

def _windows_api():
 from ctypes import wintypes
 api=ctypes.WinDLL('kernel32',use_last_error=True)
 api.CreateJobObjectW.argtypes=[ctypes.c_void_p,wintypes.LPCWSTR];api.CreateJobObjectW.restype=wintypes.HANDLE
 api.AssignProcessToJobObject.argtypes=[wintypes.HANDLE,wintypes.HANDLE];api.AssignProcessToJobObject.restype=wintypes.BOOL
 api.TerminateJobObject.argtypes=[wintypes.HANDLE,wintypes.UINT];api.TerminateJobObject.restype=wintypes.BOOL
 api.CloseHandle.argtypes=[wintypes.HANDLE];api.CloseHandle.restype=wintypes.BOOL
 return api

def spawn_owned(*args,**kwargs):
 proc=subprocess.Popen(*args,**kwargs)
 if os.name=='nt':
  api=_windows_api();handle=api.CreateJobObjectW(None,None)
  if not handle:
   proc.terminate();raise ctypes.WinError(ctypes.get_last_error())
  if not api.AssignProcessToJobObject(handle,int(proc._handle)):
   error=ctypes.get_last_error();proc.terminate();api.CloseHandle(handle);raise ctypes.WinError(error)
  proc.owned_simulator_job=handle
 return proc

def stop_owned_tree(proc):
 if proc.poll() is not None:return None
 if os.name=='nt':
  handle=getattr(proc,'owned_simulator_job',None)
  if not handle:raise RuntimeError('Refusing tree stop without an owned simulator Job')
  api=_windows_api()
  if not api.TerminateJobObject(handle,1):raise ctypes.WinError(ctypes.get_last_error())
  return dict(pid=proc.pid,returncode=0,detail='Terminated only the owned simulator Job')
 proc.terminate()
 return dict(pid=proc.pid,returncode=None,detail='Terminated spawned simulator')

def close_owned(proc):
 handle=getattr(proc,'owned_simulator_job',None)
 if handle:
  _windows_api().CloseHandle(handle);proc.owned_simulator_job=None
