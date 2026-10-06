import os
from pathlib import Path

# 模型缓存走 D 盘 + hf-mirror 镜像（必须在加载任何 HF 模型前设置）
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HOME', r'D:\86130\hf-models')

# 绕过本机 Clash 类代理：注册表 WinINET 代理会干扰 Python 的 SSL 连接，直连国内镜像更稳
os.environ['NO_PROXY'] = '*'
for _k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy'):
    os.environ[_k] = ''

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / 'data'
MATERIALS_DIR = ROOT / 'materials'
INDEX_DIR = DATA_DIR / 'faiss_index'

EMBED_MODEL = 'BAAI/bge-m3'
EMBED_DIM = 1024
TOP_K = 8
LLM_MODEL = 'deepseek-flash'
LLM_API_URL = 'https://api.deepseek.com/chat/completions'

WATERMARK_PATTERNS = ['统计战线', '就现在考研', '微信公众号']


def get_key() -> str:
    return (ROOT / '.env').read_text(encoding='utf-8').strip().split('=', 1)[1].strip()
