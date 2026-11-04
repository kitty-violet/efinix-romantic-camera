"""Bounded read-only C6/E page reader; no download or DDR access."""
import json,sys
from pathlib import Path
import read_camera_checked_snapshot as original
import read_camera_checked_snapshot_v2 as bounded
from read_camera_recovery import decode
from run_units import sha

def checked_manifest(path):
 m=original.checked_manifest(path)
 assert m['camera_checks']['recovery_mapped_cases']==20
 assert m['camera_checks']['stream_held_bits']==76
 assert m['camera_page_schema']=='C7-config/E-first-fault'
 return m

def main():
 original.decode=decode
 bounded.checked_manifest=checked_manifest
 # The child must start through this wrapper too, so its decoder is identical.
 bounded.__file__=__file__
 bounded.main()

if __name__=='__main__':main()
