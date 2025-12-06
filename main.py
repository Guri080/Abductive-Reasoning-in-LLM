import os
import google.generativeai as genai
import json
import pandas as pd
import random
from openai import OpenAI
from groq import Groq
import re

# read the key from file
API_path = ".env.txt"

API_info = {}
with open(API_path, 'r') as f:
    for line in f:
        val = line.split(':')
        API_info[val[0]] = val[1]

# load API keys
api_gemini = API_info['API_gemini'].strip()
api_groq = API_info['API_groq'].strip()

# load models
genai.configure(api_key=api_gemini)
client = Groq(api_key=api_groq)

# load JSON with dataset
with open("validation_rows.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# extract the actual rows
rows = [item["row"] for item in data["rows"]]

# convert to DataFrame
df = pd.DataFrame(rows)

# print(df.columns)

random_index = random.randint(0, len(df))

# print(df.iloc[random_index]['question'])

# load model
model = genai.GenerativeModel("gemini-2.5-flash")

# pre-processing of the {context, question, answer} to {context}
prompt = " ".join([
    "You are an assistant that rewrites questions into factual statements using the context and answer provided.",
    "Rewrite the context so that it includes the answer in a factual statement.",
    "The output should be a paragraph, not a question and do not add or remove any information.\n",
    f"{df.iloc[random_index]['question']} {df.iloc[random_index]['answer']}"
])

response = model.generate_content(f"{prompt}")

# processing {context} to {incomplete context, conclusion, missing information}
prompt = "".join([
    "Given this context generate a conclusion that encorporates the whole context",
    "Then remove one line from the context that makes the conclusion incomplete",
    "Only give me the incomplete context, conclusion, redacted line",
    "The format should be \'context:\'\'conclusion:\'\'redacted_text:\'\n"
    f"{response.text}"
])

response = model.generate_content(f"{prompt}")


def extract_context_question(text):
    '''
    test: str - The text is {incomplete context, conclusion, missing information} structure

    This method is responsible for extracting information from the text string and returns a dictionary
    with {Context, conclusion, redacted_text} keys
    '''
    pattern = r"(?mi)^(context|conclusion|redacted_text):\s*(.*?)(?=\n(?:context|conclusion|redacted_text):|$)"
    parts = dict((k.lower(), v.strip()) for k, v in re.findall(pattern, text, flags=re.DOTALL))

    return {
        "context": parts.get("context", ""),
        "conclusion": parts.get("conclusion", ""),
        "redacted_text": parts.get("redacted_text", "")
    }

raw_text = response.text

abductive_dict = extract_context_question(raw_text)

## Sanity check to see if extraction was successful
# print(abductive_dict)
# print(f"My dict keys: {abductive_dict.keys()}")

## NOTE: The keys to the abductive_dict are ['context', 'conclusion', 'redacted_text']
prompt = "".join([
    "Modified Knowledge (K')\n",
    f"{abductive_dict['context']}\n",
    "Conclusion (C)\n",
    f"{abductive_dict['conclusion']}\n",
    "Your Task\n",
    "Based on the Modified Knowledge (K') and the Conclusion (C), ",
    "what is the missing piece of information that would have originally",
    "been placed in the Knowledge to fully support and explain the conclusion?",
    "Please directly give the answer without any explanation."
])

# Prompt Model
chat_completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": f"{prompt}"}
    ]
)

response = chat_completion.choices[0].message.content

print(f"Model response:\n{response}")
print("*"*30)
print(f"Ground Truth:\n{abductive_dict['redacted_text']}")