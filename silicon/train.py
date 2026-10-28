# -*- coding: utf-8 -*-
"""LoRA 训练的公共部分（首次烘焙和夜间巩固共用）。

样本两种：
  kl   {"prompt", "response"}：教师 = 关掉 LoRA、带人格卡的基座，学生 = 开 LoRA、不带人格卡，全词表 KL（Prompt Baking）
  sft  {"messages", "response"}：多轮上下文下她的原话，对回答做交叉熵（夜间从日志学"她怎么跟你说话"）
"""
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

EOS = "<|im_end|>"


def encode(tok, messages: list, response: str, system: str | None, max_ctx: int = 1500, max_resp: int = 320):
    msgs = list(messages)
    while True:
        full = ([{"role": "system", "content": system}] if system else []) + msgs
        pre = tok(tok.apply_chat_template(full, tokenize=False, add_generation_prompt=True, enable_thinking=False),
                  add_special_tokens=False).input_ids
        if len(pre) <= max_ctx + (len(tok(system).input_ids) if system else 0) or len(msgs) <= 1:
            break
        msgs = msgs[2:] if len(msgs) > 2 else msgs[1:]
    resp = tok(response, add_special_tokens=False).input_ids
    resp = resp[:max_resp] if len(resp) > max_resp else resp + [tok.convert_tokens_to_ids(EOS)]
    return pre, resp


def logprobs(model, pre, resp, lora: bool):
    ids = torch.tensor([pre + resp], device="cuda")
    if lora:
        out = model(input_ids=ids, logits_to_keep=len(resp) + 1)
    else:
        with model.disable_adapter():
            out = model(input_ids=ids, logits_to_keep=len(resp) + 1)
    return F.log_softmax(out.logits[0, :-1].float(), -1)


def load_model(path: str, vram_gb: float, adapter: str | None, rank: int = 32):
    """加载本地模型并挂 LoRA：有 adapter 就接着训，没有就新建（秩 rank，α=2·rank，全部线性层）。"""
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(1.0, vram_gb / total))
    tok = AutoTokenizer.from_pretrained(path)
    base = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    if adapter and Path(adapter, "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base, adapter, is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(r=rank, lora_alpha=2 * rank, target_modules="all-linear", lora_dropout=0.0))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return tok, model


def train(model, tok, rows: list, persona: str, out_dir: Path, epochs: int = 2, lr: float = 1.5e-4, accum: int = 8, log=print):
    data = []
    for r in rows:
        if r["kind"] == "kl":
            msgs = [{"role": "user", "content": r["prompt"]}]
            data.append(("kl", encode(tok, msgs, r["response"], persona), encode(tok, msgs, r["response"], None)))
        else:
            data.append(("sft", None, encode(tok, r["messages"], r["response"], None)))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps = math.ceil(len(data) * epochs / accum)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    model.train()
    t0, step, run = time.time(), 0, {"kl": [], "sft": []}
    for ep in range(epochs):
        random.Random(ep).shuffle(data)
        for i, (kind, teacher, (sp, sr)) in enumerate(data):
            s_lp = logprobs(model, sp, sr, lora=True)
            if kind == "kl":
                with torch.no_grad():
                    t_lp = logprobs(model, *teacher, lora=False)
                loss = (t_lp.exp() * (t_lp - s_lp)).sum(-1).mean()
            else:
                loss = -s_lp.gather(1, torch.tensor(sr, device="cuda").unsqueeze(1)).mean()
            (loss / accum).backward()
            run[kind].append(loss.item())
            if (i + 1) % accum == 0 or i == len(data) - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    msg = {k: round(sum(v) / len(v), 4) for k, v in run.items() if v}
                    log(f"第 {ep + 1} 轮 {step}/{steps} 步 {msg} {(time.time() - t0) / 60:.1f} 分钟，"
                        f"显存峰值 {torch.cuda.max_memory_allocated() / 2**30:.2f}GB")
                    run = {"kl": [], "sft": []}
    model.save_pretrained(out_dir)
    info = {"samples": len(data), "epochs": epochs, "minutes": round((time.time() - t0) / 60, 1),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    (Path(out_dir) / "train_info.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    return info
