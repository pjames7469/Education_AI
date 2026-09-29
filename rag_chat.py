import argparse
import json
import faiss
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def main():
    # Parse command-line arguments
    p = argparse.ArgumentParser()
    p.add_argument("--question", required=True)
    p.add_argument("--use_lora", action="store_true")
    a = p.parse_args()

    # Load embedding model and FAISS index
    embed = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    idx = faiss.read_index("outputs/faiss.index")

    with open("outputs/rag_metadata.json", "r", encoding="utf-8") as f:
        docs = json.load(f)

    # Generate question embedding
    q = embed.encode(
        [a.question],
        normalize_embeddings=True
    ).astype("float32")

    # Retrieve the three most relevant documents
    scores, ids = idx.search(q, 3)

    chosen = [
        docs[i]
        for i in ids[0]
        if 0 <= i < len(docs)
    ]

    # Prepare retrieved context
    ctx = "\n".join(
        f"[{d['id']}] {d['title']}: {d['text']}"
        for d in chosen
    )

    # Build prompt
    prompt = (
        "Answer using only this context. "
        "If it is insufficient, say so. "
        "Cite source IDs in brackets.\n\n"
        f"{ctx}\n\n"
        f"Question: {a.question}"
    )

    # Load language model
    name = "Qwen/Qwen2.5-0.5B-Instruct"

    tok = AutoTokenizer.from_pretrained(name)

    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=(
            torch.float16
            if torch.cuda.is_available()
            else torch.float32
        ),
        device_map=(
            "auto"
            if torch.cuda.is_available()
            else None
        )
    )

    # Optionally load LoRA adapter
    if a.use_lora:
        model = PeftModel.from_pretrained(
            model,
            "outputs/lora_adapter"
        )

    # Format conversation using the model's chat template


    x = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )

    # Move tokenized inputs to the model's device
    x = x.to(model.device)

    # Generate answer
    with torch.no_grad():
        y = model.generate(
            **x,
            max_new_tokens=220,
            do_sample=False,
            pad_token_id=tok.eos_token_id
        )

    # Display retrieved sources
    print("\nRetrieved documents:")
    print([d["id"] for d in chosen])

    # Extract only the newly generated tokens
    input_length = x["input_ids"].shape[-1]

    answer = tok.decode(
        y[0][input_length:],
        skip_special_tokens=True
    )

    print(answer)

if __name__ == "__main__":
    main()
