import os
import google.generativeai as genai
import json
import pandas as pd
import random
from openai import OpenAI
from groq import Groq

# read the key from file
API_path = ".env.txt"

API_info = {}
with open(API_path, 'r') as f:
    for line in f:
        val = line.split(':')
        API_info[val[0]] = val[1]


api_gemini = API_info['API_gemini'].strip()
api_groq = API_info['API_groq'].strip()


# Load JSON
with open("validation_rows.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract the actual rows
rows = [item["row"] for item in data["rows"]]

# Convert to DataFrame
df = pd.DataFrame(rows)

# print(df.columns)

random_index = random.randint(0, len(df))

# print(df.iloc[random_index]['question'])

# # load model
# genai.configure(api_key=api_gemini)
# model = genai.GenerativeModel("gemini-2.5-flash")

# # pre-processing of the {context, question, answer} to {context}
# prompt = " ".join([
#     "You are an assistant that rewrites questions into factual statements using the context and answer provided.",
#     "Rewrite the context so that it includes the answer in a factual statement.",
#     "The output should be a paragraph, not a question and do not add or remove any information.\n",
#     f"{df.iloc[random_index]['question']} {df.iloc[random_index]['answer']}"
# ])

# response = model.generate_content(f"{prompt}")

# # processing {context} to {incomplete context, conclusion, missing information}
# prompt = "".join([
#     "Given this context generate a conclusion that encorporates the whole context",
#     "Then remove one line from the context that makes the conclusion incomplete",
#     "Only give me the incomplete context, conclusion, redacted line",
#     "The format should be \'Context:\'\'conclusion:\'\'redacted_text:\'\n"
#     f"{response.text}"
# ])

# response = model.generate_content(f"{prompt}")
# print(response.text)

client = Groq(api_key=api_groq)