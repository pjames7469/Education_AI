# Education LoRA + RAG learning project

This is a small, synthetic educational demo—not a classroom-validated tutor. Check factual accuracy, reading level, bias, accessibility, and rights to all data before real use.

## Workflow
1. Install Python 3.10+ and dependencies: `pip install -r requirements.txt`. Install a CUDA-compatible PyTorch build for GPU use.
2. Inspect `data/education_train.jsonl`, `education_validation.jsonl`, and `education_test.jsonl`. Each JSONL row has instruction, input, output. Add diverse, verified examples and avoid overlap between splits.
3. Fine-tune the Qwen2.5-0.5B-Instruct model using LoRA: `python train_lora.py`. Adapter saved in `outputs/lora_adapter`. First run downloads model weights.
4. Build the local vector index: `python build_index.py`. This embeds the source texts using all-MiniLM-L6-v2 and stores FAISS index + metadata in outputs.
5. Query RAG: `python rag_chat.py --question "Why do seasons happen?"`; optionally append `--use_lora`. Review retrieved source IDs and verify that the answer is supported.
6. Evaluate base and fine-tuned model: `python evaluate.py --mode base` and `python evaluate.py --mode lora`. The script reports ROUGE-L and token overlap precision/recall/F1 against held-out references. These do not prove factuality or teaching quality.
7. Peer review: use `peer_review_template.csv`. Have at least two reviewers independently score accuracy, relevance, clarity/age fit, grounding, and safety/fairness from 1–5. Record errors and disagreements.
8. Iterate: classify failures (factual, instruction, reading level, retrieval, citation), improve data or documents, retrain/reindex as needed, and rerun held-out evaluation. Never tune on the test set.

## Practical notes
The dataset is intentionally tiny to make the workflow understandable. Training may be slow on CPU. LoRA adapter weights are small relative to the base model. RAG retrieves documents at inference time; it does not update model weights. For production, add more verified sources, robust chunking, metadata filters, privacy controls, and educator review. QLoRA (4-bit quantization) is an optional memory optimization requiring compatible NVIDIA CUDA/bitsandbytes.
