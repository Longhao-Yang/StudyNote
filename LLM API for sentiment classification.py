import pandas as pd 
import numpy as np
from openai import OpenAI

df = pd.read_csv("data.csv")
info = ["cus_comment", "stars"]
df = df[info]

df = df.dropna()

def solve(x):
    return "good" if x >= 3 else "bad"  
  
df["label"] = df["stars"].apply(solve)

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="",
    base_url="",
)

def call_qwen_api(prompt: str) -> str:
    completion = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[
        {"role": "system", "content": "You are a sentiment classifier. "
                    "Output EXACTLY one word: good or bad. "
                    "No punctuation. No explanation."},
        {"role": "user", "content": prompt},
    ],
    temperature = 0,
    max_tokens = 3
    )
    return completion.choices[0].message.content 

all = 0
right = 0
for idx, row in df.iterrows():
    reply = call_qwen_api(row["cus_comment"])
    print(reply, row["label"])
    if reply == row["label"]:
        right += 1
    all += 1
acc = right / all
print(acc)

