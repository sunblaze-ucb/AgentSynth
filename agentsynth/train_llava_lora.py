#!/usr/bin/env python3
import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Fix tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Memory management optimizations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Disable for better performance

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True")

# Set matplotlib backend before any other imports to avoid GUI issues
import matplotlib
matplotlib.use('Agg')

import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

from transformers import (
    LlavaForConditionalGeneration,
    LlavaProcessor,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb


# ---------------------------
# Helpers
# ---------------------------

DEFAULT_IMAGE_TOKEN = "<image>"

def find_first_turn(conversations: List[Dict[str, str]]):
    """
    Extract (system, human, assistant) as single turn.
    If multiple turns exist, we take the FIRST human→assistant pair.
    """
    system = None
    human = None
    assistant = None

    # optional system
    if len(conversations) > 0 and conversations[0]["from"] == "system":
        system = conversations[0]["value"]

    # find first human -> assistant pair
    i = 0
    if system is not None:
        i = 1
    while i < len(conversations) - 1:
        if conversations[i]["from"] == "human" and conversations[i+1]["from"] == "gpt":
            human = conversations[i]["value"]
            assistant = conversations[i+1]["value"]
            break
        i += 1

    return system, human, assistant


def build_prompt_and_labels(tokenizer, system_text: Optional[str], human_text: str, assistant_text: str):
    """
    Compose a single-turn prompt+response with label masking.
    - We tokenize prompt (system + human) and assistant separately.
    - Labels are -100 for prompt tokens, assistant tokens as-is.
    """
    pieces = []
    if system_text:
        # Many LLaVA chat templates include system implicitly;
        # to keep things stable we place system as a prefix paragraph.
        pieces.append(system_text.strip())
    pieces.append(human_text.strip())
    prompt = "\n\n".join(pieces)

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        return_tensors="pt"
    )["input_ids"][0]

    # Assistant: do NOT add BOS/EOS here; the model/tokenizer will handle special tokens consistently
    assistant_ids = tokenizer(
        assistant_text,
        add_special_tokens=False,
        return_tensors="pt"
    )["input_ids"][0]

    input_ids = torch.cat([prompt_ids, assistant_ids], dim=0)

    labels = input_ids.clone()
    labels[: len(prompt_ids)] = -100  # mask non-assistant tokens
    return input_ids, labels


# ---------------------------
# Dataset
# ---------------------------

class LlavaChatDataset(Dataset):
    """
    Expects a JSON file with a list of records:
    {
      "id": "...",
      "images": ["images/xxx.png"],
      "conversations": [
         {"from":"system","value":"..."} (optional),
         {"from":"human","value":"<image> ..."},
         {"from":"gpt","value":"{\"action\":..., \"thoughts\":...}"}
      ]
    }
    """

    def __init__(self, data_json: str, image_root: str, processor: LlavaProcessor, tokenizer: AutoTokenizer, image_token: str = DEFAULT_IMAGE_TOKEN, image_size: int = 768, max_samples: int = None, max_length: int = 2048):
        self.items = json.load(open(data_json, "r", encoding="utf-8"))
        if max_samples is not None:
            self.items = self.items[:max_samples]
        self.image_root = image_root
        self.processor = processor
        self.tokenizer = tokenizer
        self.image_token = image_token
        self.image_size = image_size
        self.max_length = max_length

        # sanity check: ensure at least one item is valid
        assert len(self.items) > 0, "Empty dataset JSON."

    def __len__(self):
        return len(self.items)

    def _load_image(self, path_rel: str) -> Image.Image:
        path = os.path.join(self.image_root, path_rel)
        try:
            img = Image.open(path).convert("RGB")
            # Pre-resize to a standard size for faster processing
            # LLaVA typically uses 336x336 or 448x448
            target_size = (self.image_size, self.image_size)
            if img.size != target_size:
                # Use LANCZOS for older PIL versions compatibility
                try:
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                except AttributeError:
                    # Fallback for older PIL versions
                    img = img.resize(target_size, Image.LANCZOS)
            return img
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # Return a blank image as fallback
            return Image.new("RGB", (self.image_size, self.image_size), color="white")

    def __getitem__(self, idx):
        ex = self.items[idx]
        conversations = ex["conversations"]
        imgs = ex.get("images") or ex.get("image")  # support either ["..."] or "..."

        if isinstance(imgs, list):
            # Use first image (common for step-level data)
            img_rel = imgs[0]
        else:
            img_rel = imgs

        system, human, assistant = find_first_turn(conversations)
        if human is None or assistant is None:
            # Fallback/skip; Trainer will drop on collate error if needed
            raise ValueError("Sample missing human/assistant turn")

        # Replace <image> token with the processor's image token if it differs
        image_token = getattr(self.processor, "image_token", DEFAULT_IMAGE_TOKEN)
        human = human.replace(DEFAULT_IMAGE_TOKEN, image_token)

        # Load image
        image = self._load_image(img_rel)
        
        # Process both text and image together to ensure proper token alignment
        # We need to pass the full conversation text to the processor
        full_text = human  # This contains the <image> token
        processed = self.processor(
            text=full_text,
            images=image,
            return_tensors="pt",
            padding=True,
            # truncation=True,
            max_length=self.max_length,  # Use configurable max length
        )
        pixel_values = processed["pixel_values"][0]
        input_ids = processed["input_ids"][0]
        
        # Now we need to create proper labels for training
        # The processor has already handled the image tokens, so we need to mask the prompt part
        # and only train on the assistant response
        labels = input_ids.clone()
        
        # Find the assistant response tokens in the processed input_ids
        # We'll tokenize the assistant response separately to find its tokens
        assistant_tokens = self.tokenizer(
            assistant,
            add_special_tokens=False,
            return_tensors="pt"
        )["input_ids"][0]
        
        # Find where the assistant tokens appear in the full input_ids
        # This is a simplified approach - we'll look for the last occurrence of assistant tokens
        assistant_len = len(assistant_tokens)
        found_assistant = False
        
        for i in range(len(input_ids) - assistant_len + 1):
            if torch.equal(input_ids[i:i+assistant_len], assistant_tokens):
                # Found the assistant response, unmask it
                labels[i:i+assistant_len] = input_ids[i:i+assistant_len]
                found_assistant = True
                break
        
        # If we couldn't find the assistant tokens, mask everything except the last few tokens
        if not found_assistant:
            labels[:] = -100
            # Keep the last few tokens unmasked as a fallback
            labels[-min(assistant_len, len(labels)):] = input_ids[-min(assistant_len, len(labels)):]

        # Attention mask (text only; model builds vision attn internally)
        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,   # [3, H, W]
        }


# ---------------------------
# Data Collator
# ---------------------------

@dataclass
class VLMDataCollator:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: int = 8

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Clear cache before processing batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # pad text
        input_ids = [ex["input_ids"] for ex in batch]
        labels = [ex["labels"] for ex in batch]
        attn = [ex["attention_mask"] for ex in batch]
        pixel_values = [ex["pixel_values"] for ex in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attn, batch_first=True, padding_value=0
        )

        # stack images (they are already same size after processor)
        pixel_values = torch.stack(pixel_values, dim=0)

        # Optional: pad to multiple of 8 for Tensor Cores
        def _pad_to_multiple(x, multiple, pad_value):
            if x.dim() < 2:
                return x
            length = x.shape[1]
            pad_len = (multiple - (length % multiple)) % multiple
            if pad_len == 0:
                return x
            pad = torch.full((x.shape[0], pad_len), pad_value, dtype=x.dtype, device=x.device)
            return torch.cat([x, pad], dim=1)

        input_ids = _pad_to_multiple(input_ids, self.pad_to_multiple_of, self.tokenizer.pad_token_id)
        labels = _pad_to_multiple(labels, self.pad_to_multiple_of, -100)
        attention_mask = _pad_to_multiple(attention_mask, self.pad_to_multiple_of, 0)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }


# ---------------------------
# Main
# ---------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--data_json", type=str, required=True, help="Path to LLaVA-style dataset.json")
    ap.add_argument("--image_root", type=str, required=True, help="Root folder that contains image paths in JSON")
    ap.add_argument("--output_dir", type=str, default="llava_lora_out")
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--bf16", action="store_true", help="Use bfloat16")
    ap.add_argument("--fp16", action="store_true", help="Use float16")
    ap.add_argument("--packing", action="store_true", help="Pack multiple samples per sequence (usually False for VLM)")
    ap.add_argument("--seed", type=int, default=42)

    # LoRA config
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", type=str,
                    default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,mm_projector,mm_projection")
    # 4-bit
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16", choices=["float16","bfloat16","float32"])
    ap.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", choices=["nf4","fp4"])
    ap.add_argument("--bnb_4bit_use_double_quant", action="store_true")
    ap.add_argument("--max_samples", type=int, default=None, help="Limit dataset to max_samples for testing")
    ap.add_argument("--image_size", type=int, default=448, help="Image size for processing (448 or 336)")
    ap.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")

    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    compute_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.bnb_4bit_compute_dtype]

    # Load processor+tokenizer
    processor = LlavaProcessor.from_pretrained(args.model_name)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model (4-bit optional)
    if args.use_4bit:
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_name,
            torch_dtype=compute_dtype,
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
            device_map="auto",
            low_cpu_mem_usage=True,  # Reduce CPU memory usage
            # attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,  # Use flash attention if available
        )
        model = prepare_model_for_kbit_training(model)
    else:
        # full precision or bf16/fp16
        dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,  # Reduce CPU memory usage
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,  # Use flash attention if available
        )

    # Attach LoRA
    target_modules = [t.strip() for t in args.target_modules.split(",") if t.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Dataset & collator
    ds = LlavaChatDataset(
        data_json=args.data_json,
        image_root=args.image_root,
        processor=processor,
        tokenizer=tokenizer,
        image_token=getattr(processor, "image_token", DEFAULT_IMAGE_TOKEN),
        max_samples=args.max_samples,
        image_size=args.image_size,
        max_length=args.max_length,
    )
    collator = VLMDataCollator(tokenizer)

    # Training args - optimized for speed
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16 and not args.bf16,
        optim="paged_adamw_32bit" if args.use_4bit else "adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_num_workers=2,  # Increase for better data loading
        remove_unused_columns=False,  # IMPORTANT for VLM
        gradient_checkpointing=False,  # Keep disabled for speed
        seed=args.seed,
        dataloader_pin_memory=True,  # Enable for faster GPU transfer
        max_grad_norm=1.0,  # Add gradient clipping
        # Speed optimizations
        dataloader_prefetch_factor=2,  # Prefetch batches
        prediction_loss_only=True,  # Only compute loss, not other metrics
        include_inputs_for_metrics=False,  # Reduce memory usage
        # Memory optimizations
        eval_strategy="no",  # Disable evaluation for speed
        save_strategy="steps",  # Only save at specified steps
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    # If you want to save only LoRA adapters:
    if hasattr(model, "peft_config"):
        model.save_pretrained(os.path.join(args.output_dir, "lora_adapters"))


if __name__ == "__main__":
    main()
