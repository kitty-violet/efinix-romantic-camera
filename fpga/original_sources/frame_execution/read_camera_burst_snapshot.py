"""Default offline-only, manifest-bound reader for the capture-burst release."""
import read_camera_checked_snapshot as original
import read_camera_checked_snapshot_v2 as bounded
from read_camera_burst import decode
def checked_manifest(path):
 m=original.checked_manifest(path)
 assert m['camera_page_schema']=='C7-config/E-capture-progress'
 assert m['camera_checks']['capture_burst_beats']==16
 assert m['camera_checks']['capture_progress_bits']==6
 assert m['camera_checks']['stream_held_bits']==76
 return m
def main():
 original.decode=decode;bounded.checked_manifest=checked_manifest
 bounded.__file__=__file__;bounded.main()
if __name__=='__main__':main()
