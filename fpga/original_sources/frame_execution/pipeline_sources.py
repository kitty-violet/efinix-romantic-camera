"""Explicit common-netlist sources; no responders or traffic generators."""
from run_units import HERE

def rgb_pipeline_sources():
    return [HERE/n for n in (
        'rgb24_stream_pack.sv','rgb24_stream_unpack.sv',
        'rgb720_ddr_writer.sv','rgb720_ddr_reader.sv','frame_pair_owner.sv',
        'cnn_rgb_frame_pipeline.sv')]+[
        HERE.parent/'video_scaler'/n for n in (
            'axi_frame_write_dma.sv','axi_three_master_burst_arbiter.sv')]+[
        HERE.parent/'ddr_fabric'/n for n in (
            'sapphire_axi32_to_axi128.sv','ti60_ddr_four_master_fabric.sv','ddr_memory_map.vh')]
