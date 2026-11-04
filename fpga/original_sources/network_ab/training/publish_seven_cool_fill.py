"""User-requested tiny contour reduction and teacher-fitted vivid cool colors."""
import json
import shutil
import numpy as np
from . import publish_seven_structured_fill as base
from .adjust_structured_color import contours,adapt_lab
from .runtime import file_hash,write_json


def main():
    root=base.ROOT/'seven-structured-fill-cool-v2';adapter_path=root/'color_adapter.npz'
    adapter=np.load(adapter_path)['lab_chroma_affine'];original_fill=base.fill
    def colored_fill(rgb,edge,strength=.12,iterations=80):
        return adapt_lab(original_fill(rgb,edge,strength,iterations),adapter,1.15)
    base.contours=contours;base.fill=colored_fill;base.RUN_NAME='seven-structured-fill-cool-v2'
    base.PIPELINE_EXTRA=dict(contour_sigma_increment=.06,all_other_contour_parameters_unchanged=True,
        color_adapter_kind='Lab chroma affine learned from 21 TRAIN images, followed by chroma gain1.15',
        color_adapter_coefficients=adapter.tolist(),color_adapter_sha256=file_hash(adapter_path),
        color_fit_uses_validation=False,luminance_intentionally_unchanged=True,
        extra_color_operators=['RGB/Lab conversion','four-input/two-output chroma affine','chroma gain 1.15'],
        additional_source_files=['adjust_structured_color.py','fit_teacher_color_adapter.py','publish_seven_cool_fill.py'])
    base.PAGE_EXTRA='<p>这版只将轮廓尺度增加0.06，台阶样例整帧轮廓像素减少约3.7%。颜色校正从7个训练源的21张图拟合，未用展示图拟合；再小幅增加彩度。颜色校正在块面合并后执行，不重新引入原片纹理。</p>'
    base.main()
    out=root/'delivery'
    for name in ['color_adapter.npz','color_fit.json']:shutil.copy2(root/name,out/name)
    manifest=json.loads((out/'manifest.json').read_text())
    for name in ['color_adapter.npz','color_fit.json']:manifest['files'][name]=dict(bytes=(out/name).stat().st_size,sha256=file_hash(out/name))
    write_json(out/'manifest.json',manifest)


if __name__=='__main__':main()
