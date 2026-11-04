import copy
from dataclasses import asdict
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "network_ab"))
from model_contract import (ContractError, CONTRACT_PATH, SCHEMA_PATH, canonical_hash,
                            compile_contract, deployment_configuration, generated_artifacts,
                            load_contract, read_json_strict, validate_schema)
from contract_reference import (I32_MAX, I32_MIN, convolution, execute_integer, fold_batch_norm,
                                i8_to_u8, nearest2, positive_scale, q17_multiplier,
                                real_to_i8, real_to_vgg_rgb01, replicate_sample,
                                requantize, residual_value, round_half_away,
                                u8_to_i8, u8_to_real)
import model_ab_spec
import coefficient_layout


def historical_module():
    path = ROOT / "network_ab/historical/model_ab_spec_pre_contract_20260929.py"
    spec = importlib.util.spec_from_file_location("contract_historical_model", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ModelContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_contract()
        cls.plan = compile_contract(cls.contract)

    def mutate_reject(self, edit):
        changed = copy.deepcopy(self.contract)
        edit(changed)
        with self.assertRaises(ContractError):
            compile_contract(changed)

    def test_historical_source_hash_and_every_legacy_field(self):
        old = historical_module()
        self.assertEqual("be78bea79a645696038d5f685bee92793566022527d9a666e91aae41f93e3202",
                         hashlib.sha256(Path(old.__file__).read_bytes()).hexdigest())
        self.assertEqual(asdict(old.build_model_b()), asdict(model_ab_spec.build_model_b()))
        self.assertEqual(asdict(old.build_model_a()), asdict(model_ab_spec.build_model_a()))

    def test_full_shape_budget_and_training_parameter_counts(self):
        p = self.plan
        self.assertEqual((29, 24, 10256, 966758400, 11379),
                         (len(p["nodes"]), p["quantized_operator_count"], p["raw_weight_count"],
                          p["macs_per_frame"], p["trainable_parameter_count"]))
        self.assertEqual((1120, 1120, 3), (p["trainable_bn_affine_parameters"],
                                           p["bn_running_stat_values"], p["training_bias_parameters"]))
        self.assertEqual([720, 1280, 3], p["nodes"][-1]["output_shape_hwc"])
        self.assertEqual(3, p["nodes"][-1]["channels"])
        self.assertEqual([7], p["nodes"][-1]["lane_masks"])

    def test_crop_shapes_preserve_graph(self):
        for height, width in ((4, 4), (360, 640), (540, 960)):
            p = compile_contract(self.contract, input_shape=[height, width, 3])
            self.assertEqual([height, width, 3], p["nodes"][-1]["output_shape_hwc"])
            self.assertEqual(10256, p["raw_weight_count"])
        for shape in ([3, 4, 3], [4, 4, 4], [0, 4, 3]):
            with self.assertRaises(ContractError):
                compile_contract(self.contract, input_shape=shape)

    def test_explicit_bn_relu_and_skip_contract(self):
        nodes = {n["id"]: n for n in self.plan["nodes"]}
        self.assertEqual(23, sum(n["batch_norm"]["enabled"] for n in nodes.values()))
        for block in range(1, 4):
            self.assertEqual("none", nodes[f"res{block}_conv2.pw"]["activation"])
            self.assertEqual("none", nodes[f"res{block}_add"]["activation"])
            skip = "down2.pw" if block == 1 else f"res{block-1}_add"
            self.assertEqual(skip, nodes[f"res{block}_add"]["inputs"][1])
        self.assertTrue(nodes["rgb_head"]["training_bias"])
        self.assertFalse(nodes["rgb_head"]["batch_norm"]["enabled"])

    def test_legacy_exporters_consume_identical_geometry(self):
        ops = coefficient_layout.expanded_operations()
        nodes = [n for n in self.plan["nodes"] if n["quantized_index"] is not None]
        self.assertEqual([(o.name, o.input_channels, o.output_channels, o.kernel, o.stride, o.terms) for o in ops],
                         [(n["id"], n["input_channels"], n["channels"], n["kernel"], n["stride"], n["terms"]) for n in nodes])
        # The historical diagnostic image is still diagnostic; do not edit its
        # odd/even ReLU masks into a pretend trained package.
        self.assertFalse(coefficient_layout.build_layout()["complete_network_start_allowed"])
        self.assertFalse(coefficient_layout.build_layout()["residual_scaling"]["valid"])

    def test_existing_tile_schedule_is_a_checked_compatible_view(self):
        import hardware_schedule_model as schedule
        nodes={n["id"]:n for n in self.plan["nodes"]}
        visited=[]
        for group, operations, _ in schedule.build_spatial_group_definitions(320):
            for op in operations:
                for block in (range(1,4) if group=="residual" else (0,)):
                    name=op.name.replace("resX",f"res{block}")
                    row=nodes[name];visited.append(name)
                    self.assertEqual((row["input_shape_hwc"][1],row["output_shape_hwc"][1],
                                      row["input_channels"],row["channels"],row["output_shape_hwc"][0]),
                                     (op.input_width,op.output_width,op.input_channels,op.output_channels,op.output_height))
                    if op.kind!="nn2":self.assertEqual(row["stride"],op.stride)
        self.assertEqual(set(nodes)-{f"res{i}_add" for i in range(1,4)},set(visited))

    def test_schema_missing_unknown_type_and_numeric_drift_rejected(self):
        self.mutate_reject(lambda c: c["nodes"][0].pop("activation"))
        self.mutate_reject(lambda c: c["nodes"][0].update(made_up_field=1))
        self.mutate_reject(lambda c: c["nodes"][0].update(training_bias=0))
        self.mutate_reject(lambda c: c["numeric"]["requantization"].update(fraction_bits=16))
        self.mutate_reject(lambda c: c["output"].update(float_clamp=[-1, 1]))
        with self.assertRaises(ContractError):
            validate_schema(1, {"type": "integer", "unsupportedFutureKeyword": True})

    def test_graph_miswire_and_unsupported_modes_rejected(self):
        self.mutate_reject(lambda c: c["nodes"][1]["inputs"].__setitem__(0, "not_yet_defined"))
        self.mutate_reject(lambda c: c["nodes"][2].update(id="stem"))
        add = next(i for i, n in enumerate(self.contract["nodes"]) if n["id"] == "res1_add")
        self.mutate_reject(lambda c: c["nodes"][add]["inputs"].__setitem__(1, "res1_conv1.pw"))
        self.mutate_reject(lambda c: c["nodes"][1].update(channels=16))
        self.mutate_reject(lambda c: c["nodes"][0].update(padding="none"))
        self.mutate_reject(lambda c: c["nodes"][0].update(training_bias=True))

    def test_duplicate_json_and_nonfinite_values_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"invalid.json"
            for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"nested":{"a":1,"a":2}}'):
                path.write_text(text,encoding="utf-8")
                with self.assertRaises(ContractError):read_json_strict(path)

    def test_static_export_uses_contract_not_synthetic_relu(self):
        d = deployment_configuration(self.contract)
        self.assertFalse(d["complete_network_start_allowed"])
        for row, node in zip(d["operators"], [n for n in self.plan["nodes"] if n["quantized_index"] is not None]):
            self.assertEqual(row["relu_masks"], node["lane_masks"] if node["activation"] == "relu" else [0]*node["oc8_groups"])
            self.assertIsNone(row["bias_int32"])
        for residual in d["residuals"]:
            self.assertFalse(residual["valid"])
            self.assertIsNone(residual["common_scale"])
            self.assertIsNone(residual["alignment_encoding"])

    def test_artifact_regeneration_and_drift_gate(self):
        artifacts = generated_artifacts(self.contract)
        self.assertEqual(artifacts, generated_artifacts(copy.deepcopy(self.contract)))
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, "-B", str(ROOT/"network_ab/model_contract.py"), "--output-dir", directory]
            self.assertEqual(0, subprocess.run(command, capture_output=True).returncode)
            self.assertEqual(0, subprocess.run(command+["--check"], capture_output=True).returncode)
            (Path(directory)/"deployment_static.json").write_text("{}", encoding="utf-8")
            self.assertNotEqual(0, subprocess.run(command+["--check"], capture_output=True).returncode)
        changed = copy.deepcopy(self.contract)
        changed["nodes"][0]["activation"] = "none"
        self.assertNotEqual(canonical_hash(changed), self.plan["contract_sha256"])
        self.assertNotEqual(generated_artifacts(changed)["model_b_contract.vh"], artifacts["model_b_contract.vh"])

    def test_lazy_builder_does_not_import_torch(self):
        command = [sys.executable, "-B", "-c",
                   "import sys;sys.path.insert(0,'network_ab');import contract_torch;assert 'torch' not in sys.modules;print('LAZY_IMPORT_ONLY')"]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("LAZY_IMPORT_ONLY", result.stdout)
        self.assertFalse(self.plan["pytorch_execution_verified"])

    def test_namespace_package_import_keeps_legacy_api(self):
        result = subprocess.run([sys.executable,"-B","-c",
            "from network_ab import model_ab_spec,contract_reference,contract_torch;"
            "assert model_ab_spec.build_model_b().weight_count==10256;"
            "assert contract_reference.u8_to_i8(255)==127"],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(0,result.returncode,result.stderr)


class NumericalContractTests(unittest.TestCase):
    def test_all_256_rgb_values_exact_inverse_and_vgg_domain(self):
        for value in range(256):
            self.assertEqual(value, i8_to_u8(u8_to_i8(value)))
            self.assertEqual(value-128, real_to_i8(u8_to_real(value)))
            self.assertEqual(Fraction(value, 255), real_to_vgg_rgb01(u8_to_real(value)))
        self.assertEqual(255, i8_to_u8(9999))
        self.assertEqual(0, i8_to_u8(-9999))
        self.assertEqual(Fraction(1), real_to_vgg_rgb01(Fraction(127,128)))
        self.assertNotEqual(Fraction(1), (Fraction(127,128)+1)/2)

    def test_round_bias_clamp_relu_and_q17_boundaries(self):
        for n in range(-9, 10):
            self.assertEqual((abs(n)+1)//2 * (-1 if n < 0 else 1), round_half_away(Fraction(n,2)))
        self.assertEqual((1,False), requantize(1, 0, 65536))
        self.assertEqual((-1,False), requantize(-1, 0, 65536))
        self.assertEqual((0,False), requantize(-1, 0, 65536, relu=True))
        self.assertEqual((127,True), requantize(524288, I32_MAX, 1))
        self.assertEqual((-128,True), requantize(-524288, I32_MIN, 1))
        with self.assertRaises(ContractError):
            requantize(1,0,131072)
        with self.assertRaises(ContractError):
            q17_multiplier(Fraction(1),Fraction(1),Fraction(1))

    def test_replicate_stride2_coordinates(self):
        image = [[[10*y+x] for x in range(4)] for y in range(4)]
        self.assertEqual(0, replicate_sample(image,-1,-1,0))
        self.assertEqual(33, replicate_sample(image,9,9,0))
        base = {"kind":"dw3x3","channels":1,"kernel":3,"stride":2,"activation":"none"}
        # Weight2 and multiplier1/2 make samples exact. Each tap checks the
        # global 2*x,2*y center, rather than mirroring the window RTL FSM.
        for ky in range(3):
            for kx in range(3):
                weights=[[0]*9]; weights[0][ky*3+kx]=2
                result,_=convolution(image,base,weights,[0],[65536])
                expected=[[[10*max(0,min(3,2*y+ky-1))+max(0,min(3,2*x+kx-1))] for x in range(2)] for y in range(2)]
                self.assertEqual(expected,result)

    def test_conv_dw_pw_channel_order(self):
        image=[[[3,-5],[7,9]]]
        pw={"kind":"pw1x1","channels":2,"kernel":1,"stride":1,"activation":"none"}
        result,_=convolution(image,pw,[[0,2],[2,0]],[0,0],[65536]*2)
        self.assertEqual([[[-5,3],[9,7]]],result)
        dw={"kind":"dw3x3","channels":2,"kernel":3,"stride":1,"activation":"none"}
        w=[[0]*9 for _ in range(2)];w[0][4]=w[1][4]=2
        self.assertEqual(image,convolution(image,dw,w,[0,0],[65536]*2)[0])
        conv={"kind":"conv3x3","channels":1,"kernel":3,"stride":1,"activation":"none"}
        weights=[[0]*18];weights[0][4*2+1]=2
        self.assertEqual([[[-5],[9]]],convolution(image,conv,weights,[0],[65536])[0])

    def test_nearest_exact_two_by_two_no_interpolation(self):
        image=[[[1],[2]],[[3],[4]]]
        self.assertEqual([[[1],[1],[2],[2]],[[1],[1],[2],[2]],
                          [[3],[3],[4],[4]],[[3],[3],[4],[4]]],nearest2(image))

    def test_bn_fold_matches_eval_with_signed_and_zero_gamma(self):
        weights=[[2.,-3.],[-1.,4.],[3.,7.]]
        bias=[0.5,-2.,3.];gamma=[-2.,0.,1.25];beta=[0.1,3.,-4.]
        mean=[1.,-3.,2.];var=[4.,0.,0.01];eps=1e-5
        wf,bf=fold_batch_norm(weights,bias,gamma,beta,mean,var,eps)
        for x in ([0.,0.],[1.,-2.],[100.,-0.01]):
            for c in range(3):
                direct=((sum(w*v for w,v in zip(weights[c],x))+bias[c]-mean[c]) /
                        math.sqrt(var[c]+eps)*gamma[c]+beta[c])
                folded=sum(w*v for w,v in zip(wf[c],x))+bf[c]
                self.assertAlmostEqual(direct,folded,places=10)
        self.assertEqual([0.,0.],wf[1])
        with self.assertRaises(ContractError):
            fold_batch_norm(weights,None,gamma,beta,mean,[-1,0,0],eps)
        with self.assertRaises(ContractError):
            fold_batch_norm(weights,None,gamma,beta,mean,var,float("nan"))

    def test_residual_different_scales_and_single_final_quantization(self):
        p={"main_scale":[1,8],"skip_scale":[1,16],"common_scale":[1,32],"output_scale":[1,8]}
        self.assertEqual(4,residual_value(3,2,p))
        self.assertEqual(1,residual_value(0,1,p))
        self.assertEqual(-1,residual_value(0,-1,p))
        self.assertEqual(127,residual_value(127,127,p))
        self.assertEqual(-128,residual_value(-128,-128,p))
        for bad in ({}, {**p,"common_scale":None}, {**p,"main_scale":[0,1]}):
            with self.assertRaises(ContractError):
                residual_value(1,2,bad)

    def test_full_small_graph_and_missing_runtime_rejection(self):
        contract=load_contract(); plan=compile_contract(contract,input_shape=[4,4,3])
        parameters={}
        for op in plan["nodes"]:
            if op["quantized_index"] is None:continue
            weights=[[0]*op["terms"] for _ in range(op["channels"])]
            # Center/channel identity carries signed, distinguishable input
            # through all actual graph shapes; coefficients are diagnostics.
            for oc in range(op["channels"]):
                term=4 if op["kind"]=="dw3x3" else (4*op["input_channels"] if op["kernel"]==3 else 0)+oc%op["input_channels"]
                weights[oc][term]=2
            parameters[op["id"]]={"weights":weights,"bias":[0]*op["channels"],
                "multipliers":[65536]*op["channels"],"input_scale":[1,128],
                "output_scale":[1,128],"weight_scales":[[1,2] for _ in range(op["channels"])]}
        residual={f"res{b}_add":{"main_scale":[1,128],"skip_scale":[1,128],
                  "common_scale":[1,256],"output_scale":[1,128]} for b in range(1,4)}
        image=[[[1,2,3] for _ in range(4)] for _ in range(4)]
        result=execute_integer(contract,image,parameters,residual)
        self.assertEqual([[[8,16,24] for _ in range(4)] for _ in range(4)],result["output"])
        self.assertEqual(30,len(result["tensors"]))
        self.assertFalse(result["production_start_allowed"])
        with self.assertRaises(ContractError):execute_integer(contract,image,parameters,{})
        broken=copy.deepcopy(parameters);broken["stem"]["output_scale"]=None
        with self.assertRaises(ContractError):execute_integer(contract,image,broken,residual)
        broken=copy.deepcopy(parameters);broken["stem"]["multipliers"][0]=65535
        with self.assertRaises(ContractError):execute_integer(contract,image,broken,residual)


if __name__ == "__main__":
    unittest.main()
