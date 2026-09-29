
import argparse
import json
import os

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "outputs/lora_adapter"
TEST_FILE = "data/education_test.jsonl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["base", "lora"],
        default="base"
    )
    args = parser.parse_args()

    print(f"Loading model in {args.mode} mode...")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype="auto",
        device_map="auto"
    )

    if args.mode == "lora":
        model = PeftModel.from_pretrained(
            model,
            ADAPTER_PATH
        )

    model.eval()

    ds = load_dataset(
        "json",
        data_files=TEST_FILE,
        split="train"
    )

    scorer = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=True
    )

    results = []

    for row in ds:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful educational assistant. "
                    "Give clear, accurate, age-appropriate answers."
                )
            },
            {
                "role": "user",
                "content": row["instruction"] + "\n" + row["input"]
            }
        ]

        inputs = tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        )

        inputs = inputs.to(model.device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=180,
                do_sample=False,
                pad_token_id=tok.eos_token_id
            )

        input_length = inputs["input_ids"].shape[-1]

        prediction = tok.decode(
            generated[0][input_length:],
            skip_special_tokens=True
        )

        reference = row["output"]

        rouge_l = scorer.score(
            reference,
            prediction
        )["rougeL"].fmeasure

        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(reference.lower().split())

        overlap = len(pred_tokens & ref_tokens)

        precision = (
            overlap / len(pred_tokens)
            if pred_tokens else 0.0
        )

        recall = (
            overlap / len(ref_tokens)
            if ref_tokens else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        results.append({
            "instruction": row["instruction"],
            "input": row["input"],
            "reference": reference,
            "prediction": prediction,
            "rougeL": rouge_l,
            "token_precision": precision,
            "token_recall": recall,
            "token_f1": f1
        })

        print(f"\nInstruction: {row['instruction']}")
        print(f"Prediction: {prediction}")
        print(f"ROUGE-L: {rouge_l:.4f}")

    os.makedirs("outputs", exist_ok=True)

    output_file = f"outputs/eval_{args.mode}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )

    avg_rouge = (
        sum(r["rougeL"] for r in results) / len(results)
        if results else 0.0
    )

    avg_f1 = (
        sum(r["token_f1"] for r in results) / len(results)
        if results else 0.0
    )

    print("\nEvaluation complete!")
    print(f"Mode: {args.mode}")
    print(f"Average ROUGE-L: {avg_rouge:.4f}")
    print(f"Average token F1: {avg_f1:.4f}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
