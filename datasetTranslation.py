import re
import requests
import sys
import os
from openai import AzureOpenAI
import tiktoken
from dotenv import load_dotenv
import polars as pl
import pandas as pd

load_dotenv()

def translated(row):
  try:
    response = client.chat.completions.create(
        model='gpt-4o-2024-11-20',
        messages=[
            {"role": "system", "content": "You are a helpful assistant that translate query into uzbek language. Return **ONLY** translated text without your comments"},
            {"role": "user", "content": row}
        ],
        temperature=0
    )
    return response.choices[0].message.content
  except Exception as e:
    return "Error occured"

def сount_tokens(row):
  encoding = tiktoken.encoding_for_model("gpt-4")
  tokens_used = len(encoding.encode(row))
  return tokens_used

# Login using e.g. `huggingface-cli login` to access this dataset
df = pl.read_ndjson('hf://datasets/BAAI/OpenSeek-Synthetic-Reasoning-Data-Examples/CC/CC.jsonl')


client = AzureOpenAI(
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
  api_key=os.getenv("AZURE_OPENAI_KEY"),  
  api_version="2024-02-15-preview"
)



df = df.with_columns(
    pl.col("Chain-of-thought").map_elements(translated).alias("translated-Chain-of-thought")
)

df = df.with_columns(
    pl.col("instruction").map_elements(translated).alias("translated-instruction")
)

df = df.with_columns(
    pl.col("raw").map_elements(translated).alias("translated-raw")
)

df = df.with_columns(
    pl.col("translated-Chain-of-thought").map_elements(сount_tokens).alias("translated-Chain-of-thought-tokens")
)

df = df.with_columns(
    pl.col("translated-instruction").map_elements(сount_tokens).alias("translated-instruction-tokens")
)

df = df.with_columns(
    pl.col("translated-raw").map_elements(сount_tokens).alias("translated-raw-tokens")
)

df = df.with_columns(
    pl.col("Chain-of-thought").map_elements(сount_tokens).alias("Chain-of-thought-tokens")
)

df = df.with_columns(
    pl.col("instruction").map_elements(сount_tokens).alias("instruction-tokens")
)

df = df.with_columns(
    pl.col("raw").map_elements(сount_tokens).alias("raw-tokens")
)

with open("total_tokens.txt", "w") as f:
    total_thinking = df.select(pl.col("translated-Chain-of-thought-tokens").sum()).item()
    total_instruction = df.select(pl.col("instruction-tokens").sum()).item()
    total_raw = df.select(pl.col("raw-tokens").sum()).item()
    ans = (
        f"Chain of thought: {total_thinking}\n"
        f"Total Instruction: {total_instruction}\n"
        f"Total Raw: {total_raw}\n"
        f"General amount: {total_thinking + total_instruction + total_raw}"
    )
    f.write(ans)

df.to_pandas().to_csv("translated.csv", index=False)
