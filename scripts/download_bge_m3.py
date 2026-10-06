"""从 ModelScope 下载 BGE-M3 到本地（hf-mirror 大文件会 302 到被墙的 cas-bridge，故走魔搭）。"""
import os

import requests

os.environ['NO_PROXY'] = '*'

BASE = 'https://modelscope.cn/models/BAAI/bge-m3'
DEST = r'D:/86130/models/bge-m3'

# BGE-M3 模型所需文件（对齐 HF 仓库的模型文件，排除 imgs/onnx/README 等）
FILES = [
    'config.json',
    'config_sentence_transformers.json',
    'modules.json',
    'sentence_bert_config.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'special_tokens_map.json',
    'sentencepiece.bpe.model',
    '1_Pooling/config.json',
    'pytorch_model.bin',
    'colbert_linear.pt',
    'sparse_linear.pt',
]


def download(name):
    dest = os.path.join(DEST, name.replace('/', os.sep))
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f'[跳过] {name}（已存在）', flush=True)
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = f'{BASE}/resolve/master/{name}'
    with requests.get(url, stream=True, timeout=(30, 120)) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        tmp = dest + '.part'
        done = 0
        with open(tmp, 'wb') as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f'\r  {name}: {done/1e6:6.1f}/{total/1e6:.1f} MB', end='', flush=True)
        os.replace(tmp, dest)
        print(f'\n[完成] {name} ({done/1e6:.1f} MB)', flush=True)


def main():
    print(f'需下载 {len(FILES)} 个文件 -> {DEST}', flush=True)
    for f in FILES:
        try:
            download(f)
        except Exception as e:
            print(f'\n[失败] {f}: {str(e)[:120]}', flush=True)
    print('全部处理完成', flush=True)


if __name__ == '__main__':
    main()
