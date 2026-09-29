import json, os, torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
MODEL="Qwen/Qwen2.5-0.5B-Instruct"

def read(p): return [json.loads(x) for x in open(p,encoding="utf8") if x.strip()]

def fmt(x):
    q=x["instruction"] + (("\nContext: "+x["input"]) if x.get("input") else "")
    return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{x['output']}<|im_end|>"

def main():
    tok=AutoTokenizer.from_pretrained(MODEL); tok.pad_token=tok.pad_token or tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,device_map="auto" if torch.cuda.is_available() else None)
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,bias="none",task_type="CAUSAL_LM",target_modules=["q_proj","k_proj","v_proj","o_proj"]))
    model.print_trainable_parameters()
    
    def prep(batch):
        texts = []

        for instruction, input_text, output in zip(
        batch["instruction"],
        batch["input"],
        batch["output"]
        ):
            example = {
            "instruction": instruction,
            "input": input_text,
            "output": output
            }
        texts.append(fmt(example))
        # Tokenize the batch
        e = tok(texts,truncation=True,max_length=512,padding="max_length")

        e["labels"] = [ids.copy() for ids in e["input_ids"]]

        return e

        #texts=[fmt(x) for x in batch]; e=tok(texts,truncation=True,max_length=512,padding="max_length")
        
        #e["labels"]=[[t if m else -100 for t,m in zip(row,mask)] for row,mask in zip(e["input_ids"],e["attention_mask"])]
        #return e
    
    tr=Dataset.from_list(read("data/education_train.jsonl")); va=Dataset.from_list(read("data/education_validation.jsonl"))
    tr=tr.map(prep,batched=True,remove_columns=tr.column_names); va=va.map(prep,batched=True,remove_columns=va.column_names)
    args=TrainingArguments(output_dir="outputs/checkpoints",num_train_epochs=4,per_device_train_batch_size=2,gradient_accumulation_steps=4,learning_rate=2e-4,eval_strategy="epoch",save_strategy="epoch",save_total_limit=2,load_best_model_at_end=True,report_to="none",fp16=torch.cuda.is_available())
    Trainer(model=model,args=args,train_dataset=tr,eval_dataset=va,data_collator=DataCollatorForLanguageModeling(tok,mlm=False)).train()
    os.makedirs("outputs/lora_adapter",exist_ok=True); model.save_pretrained("outputs/lora_adapter"); tok.save_pretrained("outputs/lora_adapter")

if __name__=="__main__":
    main()
