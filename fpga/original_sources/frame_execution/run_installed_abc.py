"""Offline ABC command runner using the already installed Efinity library.

API reference: https://github.com/berkeley-abc/abc/blob/master/src/demo.c
This runner does not assert equivalence from a zero process exit; callers must
check the CEC result and bind both compared networks independently.
"""
import argparse,ctypes,hashlib,json,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--script',type=Path,required=True);a=p.parse_args()
root=Path('E:/fpga').resolve();s=a.script.resolve();assert s.is_relative_to(root/'tmp')
d=Path('E:/efinity-2026.1.132-windows-x64/efinity/2026.1/bin')
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
handle=os.add_dll_directory(str(d));lib=ctypes.CDLL(str(d/'libabc.dll'))
lib.Abc_Start.argtypes=[];lib.Abc_Start.restype=None
lib.Abc_Stop.argtypes=[];lib.Abc_Stop.restype=None
lib.Abc_FrameGetGlobalFrame.argtypes=[];lib.Abc_FrameGetGlobalFrame.restype=ctypes.c_void_p
lib.Cmd_CommandExecute.argtypes=[ctypes.c_void_p,ctypes.c_char_p];lib.Cmd_CommandExecute.restype=ctypes.c_int
os.chdir(s.parent);start=time.monotonic();lib.Abc_Start()
try:
 frame=lib.Abc_FrameGetGlobalFrame();assert frame
 print(json.dumps(dict(library_sha256=sha(d/'libabc.dll'),script_sha256=sha(s),runner_sha256=sha(__file__))),flush=True)
 for command in s.read_text(encoding='ascii').splitlines():
  if not command.strip() or command.startswith('#'):continue
  print('COMMAND '+command,flush=True)
  rc=lib.Cmd_CommandExecute(frame,command.encode('ascii'))
  assert rc==0,(command,rc)
finally:lib.Abc_Stop()
print(json.dumps(dict(status='COMMANDS_RETURNED_ZERO_NOT_EQUIVALENCE_ASSERTION',seconds=time.monotonic()-start)),flush=True)
