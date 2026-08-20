import gc
import os
from typing import List

import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.explanations import build_user_prompt


# Helper to tokenize a batch of prompts and generate comments in one pass
def _generate_batch(
    model,
    tokenizer,
    prompts_list: List[str],
    max_new_tokens: int,
    do_sample: bool,
) -> List[str]:
    
    encoded = tokenizer(
        prompts_list,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)

    pad_token_id = model.config.pad_token_id or tokenizer.pad_token_id or 151643

    with torch.no_grad():
        outputs = model.generate(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask'],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=pad_token_id,
            use_cache=True,
        )

    input_length = encoded['input_ids'].shape[1]
    comments = []
    for out in outputs:
        generated_tokens = out[input_length:]
        raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        comments.append(_clean_output(raw_text))
    return comments


_SYSTEM_PROMPT = (
    "Ты аналитик карточек товаров. Объясняй уже зафиксированный классификатором вердикт, "
    "не меняй его и не добавляй факты, которых нет в карточке или переданных правилах. "
    "Не выводи рассуждения, служебные теги и сам вердикт. Верни только итоговый комментарий."
)

# Helper to build chat template prompt for a single sample.
def _build_prompt(text: str, category: str, prediction: int, tokenizer) -> str:
    user_text = build_user_prompt(text, category, prediction)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]

    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt


# Strip reasoning tags and extract <rationale> content if present
def _clean_output(raw: str) -> str:
    
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if "<rationale>" in text:
        match = re.search(r"<rationale>(.*?)</rationale>", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    return text


# Load LLM model and generate comments with batched inference
def generate_comments_cuda(
    llm_model_path: str,
    df: pd.DataFrame,
    batch_size: int = 64,
    max_new_tokens: int = 120,
    do_sample: bool = False,
) -> List[str]:
    
    _is_local = os.path.exists(llm_model_path) or (
        os.path.isabs(llm_model_path) and not llm_model_path.startswith(("http://", "https://", "file://"))
    )

    gpu_cap = os.environ.get("QUALITY_LLM_MAX_MEMORY", "").strip()
    cpu_cap = os.environ.get("QUALITY_LLM_CPU_MEMORY", "16GiB").strip() or "16GiB"

    if _is_local:
        tokenizer = AutoTokenizer.from_pretrained(llm_model_path, local_files_only=True)
        load_kwargs = dict(
            torch_dtype=torch.float16,
            local_files_only=True,
            trust_remote_code=True,
            device_map="auto",
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
        load_kwargs = dict(
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
    # H100/ODS: leave unset so the 4B LLM can use the full device.
    # Local 8 GB: QUALITY_LLM_MAX_MEMORY=7GiB (see docs/mark-phase6-submit.md).
    if torch.cuda.is_available() and gpu_cap:
        load_kwargs["max_memory"] = {0: gpu_cap, "cpu": cpu_cap}
    model = AutoModelForCausalLM.from_pretrained(llm_model_path, **load_kwargs)

    model.eval()
    n = len(df)
    comments: List[str] = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_df = df.iloc[start:end]

        # Collect prompts
        prompts_list = []
        for _, row in batch_df.iterrows():
            text = row.get("text", "") or ""
            category = str(row.get("category", ""))
            prediction = 1 if row.get("pred", 0) in (1, True) else 0
            prompts_list.append(_build_prompt(text, category, prediction, tokenizer))

        # Generate in one batch call
        batch_comments = _generate_batch(
            model, tokenizer, prompts_list,
            max_new_tokens, do_sample
        )
        comments.extend(batch_comments)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return comments
