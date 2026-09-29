import argparse,json,os,torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel
from rouge_score import rouge_scorer
p=argparse.ArgumentParser(); p.add_argument("--mode",choices=["base","lora"],default="base"); a=p.parse_args()
name="Qwen/Qwen2.5-0.5B-Instruct"; tok=AutoTokenizer.from_pretrained(name); model=AutoModelForCausalLM.from_pretrained(name,torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,device_map="auto" if torch.cuda.is_available() else None)
if a.mode=="lora": model=PeftModel.from_pretrained(model,"outputs/lora_adapter")
sc=rouge_scorer.RougeScorer(["rougeL"],use_stemmer=True); rows=[json.loads(x) for x in open("data/education_test.jsonl") if x.strip()]; out=[]
for r in rows:
    prompt=r["instruction"]; x=tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=True,add_generation_prompt=True,return_tensors="pt").to(model.device)
    with torch.no_grad(): y=model.generate(x,max_new_tokens=180,do_sample=False,pad_token_id=tok.eos_token_id)
    pred=tok.decode(y[0][x.shape[-1]:],skip_special_tokens=True); ref=r["output"]; rouge=sc.score(ref,pred)["rougeL"].fmeasure
    rt=set(ref.lower().split()); pt=set(pred.lower().split()); overlap=len(rt&pt); prec=overlap/max(1,len(pt)); rec=overlap/max(1,len(rt)); f1=2*prec*rec/max(1e-9,prec+rec)
    out.append({"prompt":prompt,"reference":ref,"prediction":pred,"rougeL":rouge,"token_precision":prec,"token_recall":rec,"token_f1":f1})
os.makedirs("outputs",exist_ok=True); json.dump(out,open(f"outputs/eval_{a.mode}.json","w"),indent=2)
for k in ["rougeL","token_precision","token_recall","token_f1"]: print(k,round(sum(z[k] for z in out)/len(out),3))
print("Detailed outputs saved. Lexical metrics are imperfect proxies; use human review.")
