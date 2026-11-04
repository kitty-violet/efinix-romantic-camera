"""Package native crops, reviewed limitations, and reproducible run evidence."""
import json
from pathlib import Path
import shutil

from PIL import Image, ImageDraw, ImageFont

from .runtime import configure_workspace, file_hash, write_json


def main():
    root = configure_workspace()
    run = root / "runs/styleid-material-probe-20261002"
    font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 24)
    reviews = {
        "native720": "变化明显但拒绝：广场天空块状化、树叶网格化，公园衣服和植被重绘过度。",
        "conservative720": "减轻颜色漂移，但网格/颗粒仍存在；台阶表面出现变化，同时石材外观被过度斑驳化。",
        "clean_reference720": "换无网格建筑参考后广场树叶的网状伪影减轻，说明参考选择会影响伪影；仍有植被发白、轮廓和人物细节变化，不能作为合格监督。",
    }
    records = []
    for name, review in reviews.items():
        report = json.loads((run / name / "report.json").read_text(encoding="utf-8"))
        for row in report["records"]:
            for output in row["outputs"]:
                path = run / name / row["case"]["name"] / Path(output["file"]).name
                if file_hash(path) != output["sha256"]:
                    raise ValueError("Generated PNG hash mismatch")
                with Image.open(path) as im:
                    if im.size != (1280, 720):
                        raise ValueError("Native size changed")
                    im.verify()
        records.append({"name": name, "review": review,
                        "images": sum(len(r["outputs"]) for r in report["records"]),
                        "seconds_sum": sum(r["seconds"] for r in report["records"]),
                        "peak_gpu_allocated_mib": max(r["peak_gpu_allocated_mib"] for r in report["records"]),
                        "report": str(run / name / "report.json")})
    crops = {"quiet_stairs": (100, 380, 740, 700), "plaza_stone": (40, 400, 680, 720), "park_people": (360, 230, 1000, 550)}
    for name, crop in crops.items():
        folder = run / "clean_reference720" / name
        src = Image.open(folder / "source.png").convert("RGB")
        out = Image.open(folder / "styleid_gamma_0.75.png").convert("RGB")
        board = Image.new("RGB", (1280, 770), "#141a20")
        draw = ImageDraw.Draw(board)
        for i, (label, im) in enumerate([("原始实拍", src), ("StyleID试验 · 尚未验收", out)]):
            draw.text((i * 640 + 10, 6), label, font=font, fill="white")
            board.paste(im.resize((640, 360), Image.Resampling.LANCZOS), (i * 640, 40))
            draw.text((i * 640 + 10, 410), "下图：720p原尺度局部（未放大、未锐化）", font=font, fill="white")
            board.paste(im.crop(crop), (i * 640, 450))
        board.save(run / f"{name}_before_after.jpg", quality=97)
    for name in ["styleid_attention.py", "probe_styleid_material.py", "create_styleid_gallery.py", "finalize_styleid_probe.py"]:
        shutil.copyfile(Path(__file__).with_name(name), run / "source" / name)
    write_json(run / "decision.json", {
        "status": "VISIBLE_CHANGE_BUT_NOT_ACCEPTED_FOR_DISTILLATION",
        "selected_new_weights": None, "keep_current_model": str(root / "runs/coherent-surface-20261002/final-4850/shared_style_fp32.pt"),
        "stable_weight_sha256": file_hash(root / "runs/coherent-surface-20261002/final-4850/shared_style_fp32.pt"),
        "cases": records, "native_720p_outputs": sum(r["images"] for r in records),
        "tests": "Two GPU attention comparisons against pinned author code PASS; actual model inference PASS; generated PNG hash and size checks PASS.",
        "limits": ["Single-image probes only, no video test or flicker acceptance", "No new training, student weights or FPGA export",
                   "Validation and existing indoor holdout used for method development, not fresh blind generalization test",
                   "Joint changes in conservative trial are not a one-factor ablation; clean reference comparison changes only reference at fixed settings",
                   "MAE measures change, not quality"],
        "hardware_contract_sha256": file_hash(root.parent / "network_ab/model_b_contract.json"),
        "new_dependency_installs": False, "new_large_model_downloads": False,
        "run_bytes": sum(p.stat().st_size for p in run.rglob("*") if p.is_file()),
    })
    print(json.dumps(records, ensure_ascii=False))


if __name__ == "__main__":
    main()
