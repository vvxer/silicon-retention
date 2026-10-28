# -*- coding: utf-8 -*-
r"""配置：先读 data\config.json，再用同名环境变量（大写、前缀 SILICON_）覆盖。key 只从环境变量 DEEPSEEK_API_KEY 读。"""
import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    data_dir: str = str(ROOT / "data")
    use_lora: bool = False                          # 本地人格权重：默认不开（纯 DeepSeek + 记忆层）；开要 8GB 显卡，先运行 silicon.bake
    local_model: str = "Qwen/Qwen3.5-2B"          # 本地小模型：Hugging Face 名称或本地目录（4GB 显存可试 Qwen/Qwen3.5-0.8B，没实测）
    embed_model: str = "BAAI/bge-base-zh-v1.5"      # 检索用的向量模型（CPU 上跑）
    user_name: str = "老板"                          # 她怎么称呼你；记忆和提示里都用这个名字，要和人格卡一致
    ai_name: str = "小硅"                            # 她的名字，要和人格卡一致
    vram_gb: float = 8.0                            # 本地模型最多用多少显存
    speed: str = "medium"                           # 速度档 fast / medium / strong：越往后越像她、改写越多、越慢（见 README）
    alpha_chat: float = 1.0                         # 闲聊时人格偏移强度
    alpha_task: float = 0.5                         # 写东西、写代码等任务时的偏移强度
    retrieval_chars: int = 3000                     # 每轮检索原文最多多少字（独立预算，不被概况、习惯卡挤占）
    history_chars: int = 6000                       # 本次会话最多带多少字的历史给 DeepSeek；更早的部分靠检索取回
    max_reply_tokens: int = 800
    nightly_at: str = "03:30"                       # 聊天服务开着时每天几点自动夜间巩固；"" = 不自动（自己定时跑 silicon.nightly）
    host: str = "127.0.0.1"
    port: int = 8700
    lora_port: int = 8701                           # 本地权重服务（给 DSH 插件用，silicon.lora_server）的端口

    @property
    def data(self) -> Path:
        return Path(self.data_dir)


def load() -> Config:
    cfg = Config()
    f = Path(os.environ.get("SILICON_DATA_DIR", cfg.data_dir)) / "config.json"
    if f.exists():
        for k, v in json.loads(f.read_text(encoding="utf-8")).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    for fl in fields(Config):
        v = os.environ.get("SILICON_" + fl.name.upper())
        if v is not None:
            t = type(getattr(cfg, fl.name))
            setattr(cfg, fl.name, v.strip().lower() in ("1", "true", "yes", "on") if t is bool else t(v))
    return cfg


def save_default(cfg: Config):
    f = cfg.data / "config.json"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=1), encoding="utf-8")
