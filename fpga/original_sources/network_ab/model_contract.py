"""Single-source Model B validation and training/deployment configuration.

No torch import, trained values, production residual encoding or start permit
is fabricated. Generated files are derived artifacts bound to the JSON hash.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

DIRECTORY = Path(__file__).resolve().parent
CONTRACT_PATH = DIRECTORY / "model_b_contract.json"
SCHEMA_PATH = DIRECTORY / "model_contract.schema.json"
QUANTIZED = ("conv3x3", "dw3x3", "pw1x1")


class ContractError(ValueError):
    pass


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def read_json_strict(path: Path | str) -> dict:
    def pairs(entries):
        result = {}
        for key, value in entries:
            if key in result:
                raise ContractError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def nonfinite(value):
        raise ContractError(f"nonfinite JSON value: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=nonfinite)


def _same_json(left: Any, right: Any) -> bool:
    # Python's True == 1 must not accept an integer where a boolean is frozen.
    return canonical_hash(left) == canonical_hash(right)


def validate_schema(value: Any, rule: dict, root: dict | None = None,
                    path: str = "$") -> None:
    """Strict stdlib validator for the explicit JSON-Schema subset in v1.

    Unsupported validation keywords fail closed; the schema itself remains
    usable with ordinary draft-2020-12 tools when those become available.
    """
    root = root or rule
    supported = {"$schema", "$id", "$defs", "$ref", "title", "description",
                 "type", "const", "enum", "properties", "required",
                 "additionalProperties", "items", "minItems", "maxItems",
                 "minimum", "maximum", "minLength", "pattern"}
    unknown = set(rule) - supported
    if unknown:
        raise ContractError(f"unsupported schema keywords at {path}: {sorted(unknown)}")
    if "$ref" in rule:
        ref = rule["$ref"]
        if not ref.startswith("#/$defs/"):
            raise ContractError(f"external schema reference forbidden: {ref}")
        validate_schema(value, root["$defs"][ref.split("/")[-1]], root, path)
    if "const" in rule and not _same_json(value, rule["const"]):
        raise ContractError(f"{path}: frozen value mismatch")
    if "enum" in rule and not any(_same_json(value, v) for v in rule["enum"]):
        raise ContractError(f"{path}: not an allowed value")
    expected = rule.get("type")
    checks = {"object": lambda x: isinstance(x, dict),
              "array": lambda x: isinstance(x, list),
              "string": lambda x: isinstance(x, str),
              "boolean": lambda x: type(x) is bool,
              "integer": lambda x: type(x) is int,
              "number": lambda x: type(x) in (int, float),
              "null": lambda x: x is None}
    if expected and (expected not in checks or not checks[expected](value)):
        raise ContractError(f"{path}: expected {expected}")
    if isinstance(value, dict):
        properties = rule.get("properties", {})
        missing = set(rule.get("required", [])) - set(value)
        if missing:
            raise ContractError(f"{path}: missing {sorted(missing)}")
        if rule.get("additionalProperties") is False and set(value) - set(properties):
            raise ContractError(f"{path}: unknown properties {sorted(set(value)-set(properties))}")
        for key in value.keys() & properties.keys():
            validate_schema(value[key], properties[key], root, f"{path}.{key}")
    if isinstance(value, list):
        if len(value) < rule.get("minItems", 0) or len(value) > rule.get("maxItems", len(value)):
            raise ContractError(f"{path}: array length")
        if "items" in rule:
            for index, item in enumerate(value):
                validate_schema(item, rule["items"], root, f"{path}[{index}]")
    if isinstance(value, str):
        if len(value) < rule.get("minLength", 0):
            raise ContractError(f"{path}: short string")
        if "pattern" in rule and re.fullmatch(rule["pattern"], value) is None:
            raise ContractError(f"{path}: string pattern")
    if type(value) in (int, float):
        if not float("-inf") < value < float("inf"):
            raise ContractError(f"{path}: nonfinite number")
        if value < rule.get("minimum", value) or value > rule.get("maximum", value):
            raise ContractError(f"{path}: out of range")


def compile_contract(contract: dict, *, input_shape: list[int] | None = None,
                     check_schema: bool = True) -> dict:
    if check_schema:
        validate_schema(contract, read_json_strict(SCHEMA_PATH))
    shape = list(input_shape or contract["input"]["shape_hwc"])
    if (len(shape) != 3 or any(type(x) is not int or x <= 0 for x in shape)
            or shape[2] != 3 or shape[0] % 4 or shape[1] % 4):
        raise ContractError("input must be positive HWC RGB with H/W divisible by four")
    tensors = {"input": shape}
    rows = []
    previous = "input"
    quant_index = 0
    for index, node in enumerate(contract["nodes"]):
        name, kind = node["id"], node["kind"]
        if name in tensors or any(parent not in tensors for parent in node["inputs"]):
            raise ContractError(f"{name}: duplicate, forward or missing tensor reference")
        if node["inputs"][0] != previous:
            raise ContractError(f"{name}: main path does not follow previous tensor")
        incoming = tensors[previous]
        height, width, cin = incoming
        cout = node["channels"]
        weights = macs = terms = 0
        if kind in QUANTIZED:
            if len(node["inputs"]) != 1 or node["quantization"] != "output_int8":
                raise ContractError(f"{name}: quantized operator contract")
            k, stride = node["kernel"], node["stride"]
            if (k != (1 if kind == "pw1x1" else 3)
                    or node["padding"] != ("none" if k == 1 else "replicate_1")
                    or (kind == "pw1x1" and stride != 1)):
                raise ContractError(f"{name}: kernel/stride/padding mismatch")
            if kind == "dw3x3" and cin != cout:
                raise ContractError(f"{name}: depthwise channels changed")
            terms = k * k * (1 if kind == "dw3x3" else cin)
            if terms not in contract["hardware"]["legal_terms"]:
                raise ContractError(f"{name}: unsupported shared-array term count {terms}")
            outgoing = [(height + stride - 1) // stride, (width + stride - 1) // stride, cout]
            weights = terms * cout
            macs = outgoing[0] * outgoing[1] * weights
            if node["batch_norm"]["enabled"] and node["training_bias"]:
                raise ContractError(f"{name}: redundant train bias before BN")
            if node["activation"] == "clamp_rgb" and name != contract["output"]["node"]:
                raise ContractError("RGB Clamp must be final")
            qi = quant_index
            quant_index += 1
        else:
            qi = None
            if (node["kernel"] != 0 or node["stride"] != 1 or node["padding"] != "none"
                    or node["training_bias"] or node["batch_norm"]["enabled"]
                    or node["activation"] != "none" or cin != cout):
                raise ContractError(f"{name}: nonconvolution fields")
            if kind == "nearest_x2":
                if len(node["inputs"]) != 1 or node["quantization"] != "inherit_scale":
                    raise ContractError(f"{name}: NN preserves scale")
                outgoing = [height * 2, width * 2, cin]
            else:
                if (len(node["inputs"]) != 2 or tensors[node["inputs"][1]] != incoming
                        or node["quantization"] != "residual_int8"):
                    raise ContractError(f"{name}: residual shape/quantization mismatch")
                # The skip is the actual tensor entering the matching two-DS
                # block, never the output of an internal convolution.
                members = [r for r in rows if r["fusion_group"] == node["fusion_group"]]
                if len(members) != 4 or node["inputs"][1] != members[0]["inputs"][0]:
                    raise ContractError(f"{name}: residual skip must match block input")
                outgoing = list(incoming)
        tensors[name] = outgoing
        row = copy.deepcopy(node)
        row.update(index=index, quantized_index=qi, input_shape_hwc=list(incoming),
                   output_shape_hwc=outgoing, input_channels=cin, terms=terms,
                   raw_weights=weights, macs=macs,
                   groups=cin if kind == "dw3x3" else 1,
                   oc8_groups=(cout + 7) // 8,
                   lane_masks=[(1 << min(8, cout-i)) - 1 for i in range(0, cout, 8)])
        rows.append(row)
        previous = name
    if previous != contract["output"]["node"] or tensors[previous] != shape:
        raise ContractError("output does not restore input HWC shape")
    if quant_index != 24 or sum(n["kind"] == "residual_add" for n in rows) != 3:
        raise ContractError("expected 24 quantized ops and three residual blocks")
    if sum(n["kind"] == "nearest_x2" for n in rows) != 2:
        raise ContractError("expected two exact nearest2 stages")
    total_weights = sum(row["raw_weights"] for row in rows)
    total_macs = sum(row["macs"] for row in rows)
    if total_weights != 10256:
        raise ContractError("Model B raw weight count changed")
    if input_shape is None and total_macs != 966758400:
        raise ContractError("Model B nominal MAC count changed")
    bn_channels = sum(row["channels"] for row in rows if row["batch_norm"]["enabled"])
    train_biases = sum(row["channels"] for row in rows if row["training_bias"])
    return {"schema": "ti60-compiled-model-contract-v1", "model_id": contract["model_id"],
            "contract_sha256": canonical_hash(contract), "input_shape_hwc": shape,
            "nodes": rows, "quantized_operator_count": quant_index,
            "raw_weight_count": total_weights, "macs_per_frame": total_macs,
            "trainable_parameter_count": total_weights + 2 * bn_channels + train_biases,
            "trainable_bn_affine_parameters": 2 * bn_channels,
            "bn_running_stat_values": 2 * bn_channels,
            "training_bias_parameters": train_biases,
            "pytorch_execution_verified": False,
            "residual_alignment_rtl_verified": False}


def load_contract(path: Path | str = CONTRACT_PATH) -> dict:
    contract = read_json_strict(path)
    compile_contract(contract)
    return contract


def legacy_layer_rows(contract: dict) -> list[dict]:
    """Compatibility view consumed by existing A/B scheduling and exporters."""
    result = []
    for row in compile_contract(contract)["nodes"]:
        if result and result[-1]["name"] == row["legacy_name"]:
            old = result[-1]
            if (not row["legacy_operator"].startswith("dw3x3_s")
                    or row["kind"] != "pw1x1" or old["output"] != row["input_shape_hwc"]):
                raise ContractError("invalid legacy DS grouping")
            old["output"] = row["output_shape_hwc"]
            old["weights"] += row["raw_weights"]
            old["macs"] += row["macs"]
        else:
            if any(r["name"] == row["legacy_name"] for r in result):
                raise ContractError("noncontiguous legacy layer group")
            result.append({"name": row["legacy_name"], "operator": row["legacy_operator"],
                           "input": row["input_shape_hwc"], "output": row["output_shape_hwc"],
                           "macs": row["macs"], "weights": row["raw_weights"],
                           "note": row["legacy_note"]})
    return result


def deployment_configuration(contract: dict) -> dict:
    plan = compile_contract(contract)
    convs, residuals, graph = [], [], []
    for row in plan["nodes"]:
        graph.append({k: row[k] for k in ("index", "id", "kind", "inputs", "input_shape_hwc",
                                         "output_shape_hwc", "fusion_group", "ddr_boundary", "quantization")})
        if row["kind"] in QUANTIZED:
            convs.append({"layer_id": row["quantized_index"], "name": row["id"],
                          "kind": row["kind"], "cin": row["input_channels"], "cout": row["channels"],
                          "terms": row["terms"], "kernel": row["kernel"], "stride": row["stride"],
                          "padding": row["padding"], "oc8_groups": row["oc8_groups"],
                          "lane_masks": row["lane_masks"],
                          "relu_masks": row["lane_masks"] if row["activation"] == "relu" else [0]*row["oc8_groups"],
                          "bn_fold_required": row["batch_norm"]["enabled"],
                          "input_scale": [1, 128] if row["inputs"] == ["input"] else None,
                          "output_scale": [1, 128] if row["activation"] == "clamp_rgb" else None,
                          "weight_scales": None, "bias_int32": None, "multiplier_q17": None})
        elif row["kind"] == "residual_add":
            residuals.append({"name": row["id"], "inputs": row["inputs"],
                              "main_scale": None, "skip_scale": None, "common_scale": None,
                              "output_scale": None, "alignment_encoding": None,
                              "valid": False, "status": "OPEN_PRODUCTION_RESIDUAL_ADAPTER"})
    return {"schema": "ti60-model-static-deployment-v1", "model_id": contract["model_id"],
            "contract_sha256": plan["contract_sha256"],
            "status": "STATIC_CONFIGURATION_ONLY_PARAMETERS_UNDEFINED",
            "complete_network_start_allowed": False,
            "parameter_package_compatible_with_synthetic_v1": False,
            "graph": graph, "operators": convs, "residuals": residuals,
            "limits": {"requantization": contract["numeric"]["requantization"],
                       "residual": contract["numeric"]["residual"]}}


def generated_artifacts(contract: dict) -> dict[str, str]:
    plan, deploy = compile_contract(contract), deployment_configuration(contract)
    def js(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    lines = ["# Derived from model_b_contract.json; do not edit.\n",
             f"Contract SHA256: `{plan['contract_sha256']}`\n",
             "|id|kind|input HWC|output HWC|BN|train bias|activation|quantization|",
             "|---|---|---|---|---|---|---|---|"]
    for n in plan["nodes"]:
        lines.append(f"|{n['id']}|{n['kind']}|{n['input_shape_hwc']}|{n['output_shape_hwc']}|"
                     f"{n['batch_norm']['enabled']}|{n['training_bias']}|{n['activation']}|{n['quantization']}|")
    header = ["/* Derived static geometry, not trained parameters or a start permit. */",
              "#ifndef TI60_MODEL_B_CONTRACT_H", "#define TI60_MODEL_B_CONTRACT_H",
              "#include <stdint.h>", f'#define TI60_MODEL_CONTRACT_SHA256 "{plan["contract_sha256"]}"',
              "#define TI60_MODEL_QUANTIZED_OPS 24", "#define TI60_MODEL_COMPLETE_START_ALLOWED 0",
              "typedef struct { uint8_t kind,cin,cout,terms,kernel,stride,groups,relu[4],mask[4]; } ti60_op_static_t;",
              "static const ti60_op_static_t ti60_model_ops[24] = {"]
    verilog = ["// Derived static geometry. No trained values or start permission.",
               f"// Contract SHA256 {plan['contract_sha256']}",
               "`ifndef TI60_MODEL_B_CONTRACT_VH", "`define TI60_MODEL_B_CONTRACT_VH",
               "`define TI60_MODEL_COMPLETE_START_ALLOWED 1'b0"]
    for op in deploy["operators"]:
        rm, lm = op["relu_masks"]+[0]*(4-op["oc8_groups"]), op["lane_masks"]+[0]*(4-op["oc8_groups"])
        kind = {"conv3x3": 1, "dw3x3": 2, "pw1x1": 3}[op["kind"]]
        scalars = [kind]+[op[k] for k in ("cin", "cout", "terms", "kernel", "stride", "oc8_groups")]
        header.append("  {"+",".join(map(str,scalars))+",{"+",".join(map(str,rm))+"},{"+",".join(map(str,lm))+"}}, /* "+op["name"]+" */")
        prefix = f"`define TI60_OP_{op['layer_id']:02d}"
        verilog += [f"{prefix}_KIND 2'd{kind}", f"{prefix}_TERMS 6'd{op['terms']}", f"{prefix}_CIN 6'd{op['cin']}",
                    f"{prefix}_COUT 6'd{op['cout']}",
                    f"{prefix}_KERNEL 2'd{op['kernel']}", f"{prefix}_STRIDE 2'd{op['stride']}",
                    f"{prefix}_GROUPS 3'd{op['oc8_groups']}",
                    f"{prefix}_PAD_REPLICATE 1'b{int(op['padding']=='replicate_1')}",
                    f"{prefix}_RELU 32'h{sum(v<<(8*i) for i,v in enumerate(rm)):08x}",
                    f"{prefix}_MASK 32'h{sum(v<<(8*i) for i,v in enumerate(lm)):08x}"]
    header += ["};", "#endif"]
    verilog += ["`endif"]
    bindings = {name: hashlib.sha256((DIRECTORY/name).read_bytes()).hexdigest()
                for name in ("model_contract.schema.json", "model_contract.py",
                             "contract_reference.py", "contract_torch.py")}
    manifest = {"schema": "ti60-model-contract-generation-v1", "contract_sha256": plan["contract_sha256"],
                "contract_hash_encoding": "UTF8 canonical JSON sorted keys without whitespace",
                "source_sha256": bindings, "pytorch_execution_verified": False,
                "complete_network_start_allowed": False}
    return {"generation_manifest.json": js(manifest),
            "training_structure.json": js(plan), "deployment_static.json": js(deploy),
            "layer_table.md": "\n".join(lines)+"\n", "model_b_contract.h": "\n".join(header)+"\n",
            "model_b_contract.vh": "\n".join(verilog)+"\n"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check", action="store_true", help="reject absent or drifted generated artifacts")
    args = parser.parse_args()
    contract = load_contract(args.contract)
    outputs = generated_artifacts(contract)
    if args.check and args.output_dir is None:
        parser.error("--check requires --output-dir")
    if args.output_dir:
        if not args.check:
            args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, content in outputs.items():
            path = args.output_dir / name
            if args.check:
                if not path.is_file() or path.read_text(encoding="utf-8") != content:
                    raise ContractError(f"generated artifact absent or drifted: {path}")
            else:
                path.write_text(content, encoding="utf-8", newline="\n")
    plan = compile_contract(contract)
    print(f"CONTRACT_VALID nodes=29 quantized=24 weights={plan['raw_weight_count']} "
          f"MACs={plan['macs_per_frame']} trainable={plan['trainable_parameter_count']} "
          f"SHA256={plan['contract_sha256']}")
    print("STATIC_CONFIGURATION_ONLY; PyTorch execution UNVERIFIED; trained values UNDEFINED; complete start DENIED")


if __name__ == "__main__":
    main()
