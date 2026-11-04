"""Package the numerically verified learned-region candidate for local review."""
import argparse
import json
import zlib
from pathlib import Path
from . import b1_trained_image as codec
from .runtime import contract_guard,file_hash,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,required=True)
    ap.add_argument('--fp32-steps',type=int,default=3000);ap.add_argument('--qat-steps',type=int,default=2300);a=ap.parse_args()
    assert a.fp32_steps>=0 and a.qat_steps>=0
    contract,plan=contract_guard();p=a.package;raw=(p/'coefficients-padded.bin').read_bytes();assert len(raw)==20480
    review=json.loads((p/'review.json').read_text(encoding='utf-8'))
    assert review['integer'] and review['independent_oracle_error']==0 and all(x['export_node_error']==0 for x in review['records'])
    params,residual=codec.decode(raw);assert codec.encode(params,residual)==raw[:17792] and not any(raw[17792:])
    rejected=[]
    for name,mutate in [('truncated',lambda b:b[:-1]),('payload_crc',lambda b:b[:1900]+bytes([b[1900]^1])+b[1901:]),
                        ('nonzero_transport_padding',lambda b:b[:-1]+b'\1')]:
        try:codec.decode(mutate(raw))
        except ValueError:rejected.append(name)
        else:raise AssertionError(name)
    (p/'coefficients.bin').write_bytes(raw[:17792])
    (p/'load_words.hex').write_text(''.join(f'{int.from_bytes(raw[i:i+8],"little"):016x}\n' for i in range(0,len(raw),8)),encoding='ascii')
    back=b''.join(int(line,16).to_bytes(8,'little') for line in (p/'load_words.hex').read_text().splitlines());assert back==raw
    write_json(p/'verification.json',dict(status='NUMERICALLY_VERIFIED_VISUAL_RESEARCH_CANDIDATE',
        bytes=17792,transport_bytes=20480,sha256=file_hash(p/'coefficients-padded.bin'),crc32=zlib.crc32(raw),
        graph=plan['model_id'],contract_sha256=file_hash(Path(__file__).parents[1]/'model_b_contract.json'),
        learned_steps=a.fp32_steps+a.qat_steps,fp32_steps=a.fp32_steps,qat_steps=a.qat_steps,convolutions=len(params),nodes=len(plan['nodes']),
        independent_oracle_error=0,export_all_node_error=0,rejected=rejected,hex_little_endian_64bit_roundtrip=True,
        fully_learned=False,training='Encoder and region/chroma predictor learned; RGB head is fixed colour decoding',
        postfilter=False,actual_board=False,visual_accepted=False,short_rtl=False))
    (p/'README.txt').write_text('B1颜色分区学习研究候选\n\n17792字节有效数据，20480字节传输镜像，load_words.hex为2560个小端64位字。\n'
        '编码器和区域/色度预测器经过真实训练，颜色转换头是固定系数。全部运算在现有B1图内；没有外部后处理。\n'
        '本包通过数值复演，不代表已经完成画质验收或真实上板。图像轮廓、人物和稳定性需要结合本版视频审查。\n'
        '不要覆盖当前硬件的启动镜像，也不要把本候选标记为已放行风格包。\n',encoding='utf-8')
    print('REGION_PACKAGE_VERIFIED',len(raw),file_hash(p/'coefficients-padded.bin'),flush=True)


if __name__=='__main__':main()
