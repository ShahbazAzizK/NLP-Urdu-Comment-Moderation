"""Optional no-cloud LLM evaluation using a locally running Ollama server.

Start Ollama, pull a model, then run:
  python testing/ollama_prompt_eval.py --model qwen3.5:4b --n 30
Predictions append to JSONL after every request; reruns resume without repeats.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,time
from pathlib import Path
import requests
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/2020paper/roman_urdu_hate_speech-main'
OUT=Path(__file__).resolve().parent/'results'
LABEL={0:'Abusive',1:'Not abusive'}
# Reviewed common lexical correspondences; mappings are evidence for interpretation,
# never class labels. Expand only with native-speaker review.
ALIGN=[('nahi','????'),('nhi','????'),('nai','????'),('kyun','????'),('acha','????'),
       ('mujhe','????'),('mujhy','????'),('tum','??'),('hai','??'),('kuch','???'),
       ('bohat','???'),('aap','??'),('ap','??'),('dost','????'),('ghar','???')]

def read(name):
 with (DATA/f'task_1_{name}.tsv').open(encoding='utf-8-sig',newline='',errors='replace') as f:
  return [(r[0].strip(),int(r[1])) for r in csv.reader(f,delimiter='\t') if len(r)==2 and r[0].strip() and r[1] in {'0','1'}]
def digest(row): return hashlib.sha256((row[0]+'\t'+str(row[1])).encode('utf-8')).hexdigest()
def select(rows,n):
 # balanced deterministic selection; eliminate train-test exact-text leakage
 train=read('train'); evaltexts={t.casefold() for t,_ in read('validation')+rows}
 train=[r for r in train if r[0].casefold() not in evaltexts]
 demos=[]
 for lab in (0,1): demos.extend(sorted((r for r in train if r[1]==lab),key=digest)[:2])
 chosen=[]
 for lab in (0,1): chosen.extend(sorted((r for r in rows if r[1]==lab),key=digest)[:(n+1)//2 if lab==0 else n//2])
 return chosen,demos

def prompt(text,condition,demos):
 task=('You classify Roman Urdu social-media comments. Label Abusive if it contains abuse, hate, insults, or profanity; otherwise label Not abusive. Consider meaning, not spelling. Return exactly one label: Abusive or Not abusive.\n')
 shots='\n'.join(f'Comment: {t}\nLabel: {LABEL[y]}' for t,y in demos)
 align='\n'.join(f'{r} = {u}' for r,u in ALIGN[:int(condition.rsplit("n",1)[1])]) if condition.startswith('alignment_n') else ''
 if condition=='zero_shot': return task+f'Comment: {text}\nLabel:'
 body=task+('Examples:\n'+shots+'\n\n' if shots else '')
 if align: body+='Roman Urdu to Urdu-script lexical mappings (use only to interpret spelling):\n'+align+'\n\n'
 return body+f'Comment: {text}\nLabel:'
def parse(s):
 t=s.strip().lower()
 if t in {'abusive','offensive'}: return 0
 if t in {'not abusive','normal','not offensive'}: return 1
 import re
 m=re.search(r'\b(not\s+abusive|not\s+offensive|abusive|offensive|normal)\b',t)
 if not m: raise ValueError('unrecognized label')
 return 1 if m.group(1).startswith('not ') or m.group(1)=='normal' else 0

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',default='qwen3.5:4b');ap.add_argument('--n',type=int,default=30);ap.add_argument('--host',default='http://localhost:11434');args=ap.parse_args()
 if args.n<2: ap.error('--n must be >=2')
 test=read('test'); selected,demos=select(test,args.n)
 out=OUT/'ollama_prompt_predictions.jsonl'; out.parent.mkdir(parents=True,exist_ok=True)
 conditions=['zero_shot','few_shot','alignment_n5','alignment_n15']
 existing={}
 if out.exists():
  for line in out.read_text(encoding='utf-8').splitlines():
   try:
    r=json.loads(line); existing[(r['example_hash'],r['condition'],r['model'])]=r
   except Exception: pass
 total=len(selected)*len(conditions); done=0
 with out.open('a',encoding='utf-8') as f:
  for i,(text,gold) in enumerate(selected):
   h=digest((text,gold))
   for cond in conditions:
    key=(h,cond,args.model)
    if key in existing: done+=1; continue
    record={'example_hash':h,'row_id':i,'text':text,'gold':gold,'gold_name':LABEL[gold],'condition':cond,'model':args.model}
    start=time.perf_counter()
    try:
     res=requests.post(args.host.rstrip('/')+'/api/chat',json={'model':args.model,'messages':[{'role':'user','content':prompt(text,cond,demos)}],'stream':False,'options':{'temperature':0,'num_predict':12}},timeout=300)
     if not res.ok: raise RuntimeError(f'HTTP {res.status_code}: {res.text[:300]}')
     answer=res.json()['message']['content']; record['raw_response']=answer; record['prediction']=parse(answer)
    except Exception as e: record['error']=f'{type(e).__name__}: {str(e)[:300]}'
    record['latency_seconds']=time.perf_counter()-start
    f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();done+=1
    print(f'{done}/{total}: {cond} row {i+1}',flush=True)
 allrows=[json.loads(x) for x in out.read_text(encoding='utf-8').splitlines() if x.strip()]
 summary={'dataset':'RUHSOLD Task 1 provided test split','n_examples':len(selected),'model':args.model,'temperature':0,'conditions':{}}
 for cond in conditions:
  rr=[r for r in allrows if r.get('model')==args.model and r.get('condition')==cond and r.get('example_hash') in {digest(x) for x in selected}]
  ok=[r for r in rr if 'prediction' in r]
  summary['conditions'][cond]={'attempted':len(rr),'valid':len(ok),'failed':len(rr)-len(ok)}
  if ok:
   rep=classification_report([r['gold'] for r in ok],[r['prediction'] for r in ok],labels=[0,1],target_names=[LABEL[0],LABEL[1]],output_dict=True,zero_division=0)
   summary['conditions'][cond].update({'coverage':len(ok)/len(rr),'accuracy':accuracy_score([r['gold'] for r in ok],[r['prediction'] for r in ok]),'macro_f1':rep['macro avg']['f1-score'],'class_report':rep,'confusion_matrix':confusion_matrix([r['gold'] for r in ok],[r['prediction'] for r in ok],labels=[0,1]).tolist(),'mean_latency_seconds':sum(r['latency_seconds'] for r in ok)/len(ok)})
 (OUT/'ollama_prompt_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
