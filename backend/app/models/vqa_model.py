"""
Real Remote-Sensing Visual Question Answering (VQA) Model.

Model: HuggingFaceTB/SmolVLM-256M-Instruct
Architecture: Idefics3 Vision-Language Model (~256M parameters)
Features: Multimodal image-text instruction tuning with real token-probability confidence.
"""

import os
import logging
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

logger = logging.getLogger(__name__)

VQA_MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"


def load_vqa_model(device: str = "cpu") -> Tuple[AutoProcessor, AutoModelForImageTextToText]:
    """
    Loads SmolVLM-256M-Instruct processor and model from Hugging Face Hub.
    """
    logger.info(f"Loading SmolVLM VQA model ({VQA_MODEL_ID}) on device='{device}'...")
    processor = AutoProcessor.from_pretrained(VQA_MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        VQA_MODEL_ID,
        dtype=torch.float32
    )
    model.to(device)
    model.eval()
    logger.info("SmolVLM VQA model loaded successfully and ready for evaluation.")
    return processor, model


def run_vqa_inference(
    model_tuple: Tuple[AutoProcessor, AutoModelForImageTextToText],
    image_path: str,
    query: str,
    device: str = "cpu",
    max_new_tokens: int = 45
) -> Dict[str, Any]:
    """
    Executes real multimodal VLM inference on satellite imagery.
    Computes exact mathematical confidence from generated token softmax probabilities.
    """
    processor, model = model_tuple
    image = Image.open(image_path).convert("RGB")

    # Resize large rasters to max 384x384 for fast CPU attention while maintaining visual fidelity
    max_dim = 384
    if max(image.size) > max_dim:
        image.thumbnail((max_dim, max_dim), Image.Resampling.BILINEAR)

    # Format remote sensing query prompt
    clean_query = query.strip()
    if not clean_query.endswith("?"):
        clean_query += "?"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text", 
                    "text": f"Satellite imagery analysis: {clean_query} Provide a direct, concise, factual remote-sensing answer."
                }
            ]
        }
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        generation_output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
            temperature=None,
            top_p=None
        )

    # Extract generated tokens
    gen_tokens = generation_output.sequences[0, input_len:]
    answer = processor.decode(gen_tokens, skip_special_tokens=True).strip()

    # Clean any assistant prefix artifacts
    if "Assistant:" in answer:
        answer = answer.split("Assistant:")[-1].strip()

    # Calculate exact mathematical confidence from softmax probabilities of generated tokens
    token_probs = []
    if hasattr(generation_output, "scores") and generation_output.scores:
        for idx, score_tensor in enumerate(generation_output.scores):
            if idx < len(gen_tokens):
                token_id = gen_tokens[idx]
                prob = torch.softmax(score_tensor[0], dim=-1)[token_id].item()
                token_probs.append(prob)

    if token_probs:
        confidence = round(float(sum(token_probs) / len(token_probs)), 4)
    else:
        confidence = None

    return {
        "answer": answer,
        "confidence": confidence,
        "tokens_generated": len(gen_tokens),
        "model_name": "SmolVLM-256M-Instruct (Multimodal VQA)",
        "backend": "real"
    }
