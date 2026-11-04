"""Freeze the user's accepted A without changing weights or historical evidence."""
import argparse, hashlib, json, re, shutil, sys
from pathlib import Path
from urllib.parse import unquote, urlsplit
from .b1_trained_image import decode
from .runtime import file_hash, write_json, contract_guard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); src = a.source.resolve(); dst = a.out.resolve()
    assert not dst.exists(); dst.mkdir(parents=True)
    raw = (src/'delivery/coefficients-padded.bin').read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == 'ceebe7b3187fd47fbf0924883d19a9a335923d9de378ae09d30f362d9b9d0523'
    parameters, residual = decode(raw); contract, plan = contract_guard()
    for name in ('coefficients-padded.bin','load_words.hex','selected-A.pt','review.json','selection.json','selected-comparison.png','video-motion-summary.json','delivery-validation.json'):
        shutil.copy2(src/'delivery'/name, dst/name)
    (dst/'coefficients.bin').write_bytes(raw[:17792])
    (dst/'assets').mkdir(); copied = {}; text = (src/'delivery/watch.html').read_text('utf-8')
    def replace(match):
        attr, value = match.group(1), match.group(2)
        if urlsplit(value).scheme: return match.group(0)
        source = (src/'delivery'/unquote(value)).resolve(); assert source.is_file(), source
        if source.parent == src/'delivery' and (dst/source.name).is_file(): target = source.name
        else:
            key = str(source)
            if key not in copied:
                name = hashlib.sha256(key.encode()).hexdigest()[:10]+'_'+source.name
                shutil.copy2(source,dst/'assets'/name); copied[key] = 'assets/'+name
            target = copied[key]
        return f'{attr}="{target}"'
    text = re.sub(r'(src|href|poster)="([^"]+)"', replace, text)
    text = text.replace('B1 色块风格 · 三版中选 A','B1 A · 用户已接受的当前结果')
    text = text.replace('<h1>', '<p>2026-10-08：用户明确接受 A 作为当前结果。此确认是画面结果验收，未代表 FPGA 实板已验证。</p><h1>', 1)
    (dst/'watch.html').write_text(text,encoding='utf-8')
    (dst/'evidence').mkdir()
    for name,path in {'three-revisions.json':src/'review.json','edges.json':src/'edge-audit/results.json','narrow.json':src/'narrow-audit/results.json','short-motion.json':src/'motion-audit/review.json','full-videos.json':src/'selected-A-videos/results.json','reserved.json':src/'reserved-A-review/review.json'}.items():
        shutil.copy2(path,dst/'evidence'/name)
    project = Path(__file__).resolve().parents[2]
    (dst/'contract').mkdir()
    for name in ('model_b_contract.json','model_contract.schema.json'):
        shutil.copy2(project/'network_ab'/name,dst/'contract'/name)
    sys.path.insert(0,str(project/'network_ab'))
    from coefficient_layout import build_layout
    write_json(dst/'contract/coefficient_layout.json',build_layout())
    write_json(dst/'contract/canonical_graph.json',plan)
    write_json(dst/'residual_encoding.json',residual)
    write_json(dst/'acceptance.json',dict(version='B1-A-accepted-20261008',accepted_by='user',accepted_on='2026-10-08',scope='current_visual_result',user_statement='OK，我们先拿 A 作为结果，A 的结果其实可以了。',package_sha256=digest,transport_bytes=20480,effective_bytes=17792,le64_words=2560,actual_board=False,new_hardware_release=False,geometry='manually_constructed_inside_unchanged_B1',colour_head='calibrated_on_21_TRAIN_frames',historical_review_preserved=True,contract_file_sha256=file_hash(dst/'contract/model_b_contract.json'),canonical_contract_sha256=plan['contract_sha256']))
    write_json(dst/'asset-origins.json',copied)
    print('ACCEPTED_FREEZE',dst,digest,len(copied),flush=True)


if __name__ == '__main__': main()
