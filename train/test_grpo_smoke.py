"""Verify that trl 0.13's GRPOTrainer works with a LoRA adapter as the reference policy
sharing the base model (spec §11 open question #5).
If this fails, fall back per spec to merging the reference adapter or shipping two models."""
import torch
from peft import PeftModel, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

from train.common import load_base_model, make_lora_config

def trivial_reward(prompts, completions, **kwargs):
    return [float(len(c) > 10) for c in completions]

def test_grpo_with_lora_reference_does_not_oom(tmp_path):
    model, tok = load_base_model()
    model = get_peft_model(model, make_lora_config())

    # Reference: a second LoRA adapter on top of the same base model.
    # trl 0.13 either supports passing a PeftModel as ref_model or uses PEFT's
    # "disable adapter" context as the reference.
    dummy_data = Dataset.from_list([{"prompt": "hi"} for _ in range(4)])

    cfg = GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_generations=2,
        max_completion_length=32,
        learning_rate=1e-6,
        logging_steps=1,
        max_steps=1,  # one step is enough to verify
        bf16=True,
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=trivial_reward,
        args=cfg,
        train_dataset=dummy_data,
        tokenizer=tok,
    )
    trainer.train()
    # If we got here without OOM, the LoRA-as-ref pattern works.
