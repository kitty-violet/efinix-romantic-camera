"""Use the released mapped circuit to avoid reoptimizing unrelated RTL.

Official synthesis still creates a fresh VDB. The map is not used directly as
a physical database, and the old implementation is never overwritten.
"""
from pathlib import Path
import prepare_video_sensor72 as prepare
from run_units import ROOT,sha

if __name__=='__main__':
    prepare.NATIVE=ROOT/'tmp/frame-execution/video-input-bit2/outflow/ti60f225_oob.map.v'
    prepare.NATIVE_SHA='21bfdc8a33626feed493cddc74fd73f7c191c7ad33346c2c520f6a6582407c26'
    assert sha(prepare.NATIVE)==prepare.NATIVE_SHA
    prepare.main()
