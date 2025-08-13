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
  response = client.chat.completions.create(
      model='gpt-4o-2024-11-20',
      messages=[
          {"role": "system", "content": "You are a helpful assistant that translate query into uzbek language. Return **ONLY** translated text without your comments"},
          {"role": "user", "content": row}
      ],
      temperature=0
  )
  return response.choices[0].message.content

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
    pl.col("text").map_elements(translated).alias("translated-text")
)

df.to_pandas().to_excel("translatedFull.xlsx", index=False)

# def annotate(df):
    



# print(response.choices[0].message.content)

# # Optional: Count tokens
# encoding = tiktoken.encoding_for_model("gpt-4")
# tokens_used = len(encoding.encode(response.choices[0].message.content))
# print(f"Tokens in output: {tokens_used}")