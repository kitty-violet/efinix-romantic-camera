"""Cold-start preview prerequisite: stage/read back only the accepted 20 KiB A.

No LOAD/LIVE, mailbox writes, bitstream download, or Flash access. Existing
frozen deployment and transport dependencies are validated before USB opens.
"""
import argparse, hashlib, json, subprocess, sys, time, zlib
from pathlib import Path
from run_units import ROOT, sha
from video_service_board import bound
from fixed_jtag_loader import WINDOWS, validate_payload
from accepted_a_ddr_stage import StageBus, metadata
from ddr_scratch_bringup import READ_STATUS, guard, exercise, ScratchBus
from windows_process_tree_safe import stop_launched_tree

EXPECTED = 'ceebe7b3187fd47fbf0924883d19a9a335923d9de378ae09d30f362d9b9d0523'
BASE, SIZE = WINDOWS['coefficients-padded']


class CoefficientBus(StageBus):
    def access(self, writing, address, value=0):
        if not (BASE <= address < BASE+SIZE or (not writing and address in READ_STATUS)):
            raise ValueError('Coefficient-only address policy')
        return super().access(writing, address, value)


def transfer_coefficients(bus, image, out, save):
    if len(image) != SIZE or hashlib.sha256(image).hexdigest() != EXPECTED:
        raise ValueError('Only immutable accepted A padded package is allowed')
    if any(image[17792:]) or zlib.crc32(image) != 0x7c6149bf:
        raise ValueError('Padding/CRC mismatch')
    out.update(status='CHECKING_IDLE', expected=metadata(image), guards=[],
               written_bytes=0, verified_bytes=0, mismatch_count=0,
               cnn_started=False, flash_written=False, mailbox_writes=0)
    save()
    out['guards'].append(guard(bus)); save()
    out['status']='WRITING_COEFFICIENTS'; save()
    for off in range(0,SIZE,4):
        bus.write(BASE+off,int.from_bytes(image[off:off+4],'little'))
        out['written_bytes']=off+4
        if (off+4)%4096==0:save()
    out['guards'].append(guard(bus));out['status']='READING_ALL_COEFFICIENTS';save()
    actual=bytearray()
    for off in range(0,SIZE,4):
        word=bus.read(BASE+off).to_bytes(4,'little');actual.extend(word)
        out['verified_bytes']=off+4
        if word!=image[off:off+4]:
            out.update(status='READBACK_MISMATCH',mismatch_count=1,
                       first_mismatch=dict(address=f'{BASE+off:08x}',
                       expected=image[off:off+4].hex(),actual=word.hex()))
            save();raise IOError('Coefficient readback mismatch; no retry')
        if (off+4)%4096==0:save()
    out['readback']=metadata(actual)
    if out['readback']!=out['expected']:raise IOError('Full SHA/CRC mismatch')
    out['guards'].append(guard(bus))
    out['status']='COEFFICIENTS_DDR_VERIFIED_NOT_RESIDENT_NOT_RUNNING';save()


def worker(a,m,r):
    data=validate_payload(Path(m['a_integration']))['coefficients-padded']
    if data!=(a.manifest.parent/'coefficients-padded.bin').read_bytes():
        raise ValueError('Frozen coefficient payload changed')
    out=dict(status='CONNECTING',manifest_sha256=sha(a.manifest),
             program_receipt_sha256=sha(a.program_receipt),helper_sha256=sha(__file__),
             tests_sha256=sha(a.test_receipt),hardware_access=False,
             scratch=dict(patterns=[],guards=[]),coefficients={},
             cnn_started=False,flash_written=False,compute_dma_ddr_verified=False)
    def save():a.report.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    save();link=None
    try:
        from fixed_jtag_physical_b import PhysicalBTransport
        from stable_jtag_reply import StableReplyTransport
        link=PhysicalBTransport(r['url']);stable=StableReplyTransport(link)
        out['hardware_access']=True
        sb=ScratchBus(stable,seconds=60);exercise(sb,out['scratch'],save)
        out['scratch_transport']=dict(transactions=sb.transactions,errors=sb.errors)
        cb=CoefficientBus(stable,seconds=110)
        try:transfer_coefficients(cb,data,out['coefficients'],save)
        finally:
            out['coefficient_transport']=dict(transactions=cb.transactions,errors=cb.errors,
                                            writes=cb.writes,reads=cb.reads);save()
        out['status']='PREVIEW_COEFFICIENT_DDR_READY_NO_LOAD_OR_RUN'
    except Exception as exc:out.update(status='FAIL_PREFIX_SAVED',error=repr(exc));raise
    finally:
        if link:link.close()
        save()


def main():
    p=argparse.ArgumentParser()
    for n in ('manifest','program-receipt','report','test-receipt'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--supervised',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);a=p.parse_args()
    for n in ('manifest','program_receipt','report','test_receipt'):setattr(a,n,getattr(a,n).resolve())
    m,r=bound(a.manifest,a.program_receipt)
    proof=json.loads(a.test_receipt.read_text(encoding='utf-8'))
    if proof['status']!='PASS' or proof['tests']<6 or proof['helper_sha256']!=sha(__file__):
        raise ValueError('Current focused tests required')
    if sha(Path(__file__).with_name('test_video_service_stage_preview_coefficients.py'))!=proof['test_sha256']:
        raise ValueError('Test file changed')
    if not a.execute:print('PREVIEW_COEFFICIENT_STAGE_OFFLINE_BOUND_NO_USB');return
    if not a.supervised or a.report.exists() or not a.report.is_relative_to(ROOT/'tmp/board-test'):
        raise ValueError('New supervised board-test report required')
    if a.worker:worker(a,m,r);return
    cmd=[sys.executable,'-u',__file__,'--manifest',str(a.manifest),'--program-receipt',str(a.program_receipt),
         '--report',str(a.report),'--test-receipt',str(a.test_receipt),'--execute','--supervised','--worker']
    with a.report.with_suffix('.log').open('x') as log:
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);start=time.monotonic()
        try:rc=child.wait(timeout=200);host=dict(exit_code=rc,seconds=time.monotonic()-start)
        except subprocess.TimeoutExpired:rc=124;host=dict(exit_code=rc,cleanup=stop_launched_tree(child))
    a.report.with_suffix('.host.json').write_text(json.dumps(host,indent=2),encoding='utf-8')
    print(json.dumps(host));raise SystemExit(rc)


if __name__=='__main__':main()
