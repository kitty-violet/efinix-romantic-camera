"""Bounded official-map feasibility for identical local LUT/FF drivers."""
import argparse,json,traceback
from pathlib import Path
import probe_native_map_roundtrip as mapping
from run_units import ROOT,sha
from audit_snapshot_cells import cells
from probe_full_native_split_map import params

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve()
    assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
    (root/'runner.py').write_bytes(Path(__file__).read_bytes());mapping.RUN=root
    src='''module native_roundtrip(input clk, ce, rst, input [3:0] data, output [5:0] result);
(* syn_keep="true" *) wire a,b,c,q0,q1,q2;
EFX_LUT4 #(.LUTMASK(16'h4040)) dec0(.I0(data[0]),.I1(data[1]),.I2(data[2]),.I3(data[3]),.O(a));
EFX_LUT4 #(.LUTMASK(16'h4040)) dec1(.I0(data[0]),.I1(data[1]),.I2(data[2]),.I3(data[3]),.O(b));
EFX_LUT4 #(.LUTMASK(16'h4040)) dec2(.I0(data[0]),.I1(data[1]),.I2(data[2]),.I3(data[3]),.O(c));
EFX_FF #(.SR_SYNC(1),.SR_VALUE(0)) ff0(.D(data[0]),.CLK(clk),.CE(ce),.SR(rst),.Q(q0));
EFX_FF #(.SR_SYNC(1),.SR_VALUE(0)) ff1(.D(data[0]),.CLK(clk),.CE(ce),.SR(rst),.Q(q1));
EFX_FF #(.SR_SYNC(1),.SR_VALUE(0)) ff2(.D(data[0]),.CLK(clk),.CE(ce),.SR(rst),.Q(q2));
EFX_FF #(.SR_SYNC(1),.SR_VALUE(0)) out0(.D(a),.CLK(clk),.CE(ce),.SR(rst),.Q(result[0]));
EFX_FF #(.SR_SYNC(1),.SR_VALUE(0)) out1(.D(b),.CLK(clk),.CE(ce),.SR(rst),.Q(result[1]));
EFX_FF #(.SR_SYNC(1),.SR_VALUE(0)) out2(.D(c),.CLK(clk),.CE(ce),.SR(rst),.Q(result[2]));
EFX_LUT4 #(.LUTMASK(16'h6666)) out3(.I0(q0),.I1(data[1]),.I2(1'b0),.I3(1'b0),.O(result[3]));
EFX_LUT4 #(.LUTMASK(16'h6666)) out4(.I0(q1),.I1(data[2]),.I2(1'b0),.I3(1'b0),.O(result[4]));
EFX_LUT4 #(.LUTMASK(16'h6666)) out5(.I0(q2),.I1(data[3]),.I2(1'b0),.I3(1'b0),.O(result[5]));
endmodule
'''
    try:
        stage,mapped=mapping.map_stage('kept',src)
        c=cells(mapped);ps=params(mapped)
        for name in ('dec0','dec1','dec2'):assert c[name][0]=='EFX_LUT4'
        for name in ('ff0','ff1','ff2'):assert c[name][0]=='EFX_FF'
        assert len({c[n][1]['O'] for n in ('dec0','dec1','dec2')})==3
        assert len({c[n][1]['Q'] for n in ('ff0','ff1','ff2')})==3
        for group,output in ((('dec0','dec1','dec2'),'O'),(('ff0','ff1','ff2'),'Q')):
            ref=group[0]
            for n in group[1:]:
                assert {p:v for p,v in c[n][1].items() if p!=output}=={p:v for p,v in c[ref][1].items() if p!=output}
                assert ps[n]==ps[ref]
        result=dict(status='OFFICIAL_MAP_LOCAL_IDENTICAL_DRIVERS_RETAINED',stage=stage,
            runner_sha256=sha(__file__),physical_pass=False,hardware_access=False)
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    except Exception as e:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise

if __name__=='__main__':main()
