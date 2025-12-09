import os
import google.generativeai as genai
import json
import pandas as pd
import random
from openai import OpenAI
from groq import Groq
import re

import csv
import time

# read the key from file
API_path = ".env.txt"

API_info = {}
with open(API_path, "r") as f:
    for line in f:
        val = line.split(":")
        API_info[val[0]] = val[1]

# load API keys
api_gemini = API_info["API_gemini"].strip()
api_groq = API_info["API_groq"].strip()

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

model = genai.GenerativeModel("gemini-2.5-flash")


def extract_parts(text):
    pattern = r"(?mi)^ *(context|conclusion|redacted_text):\s*(.*?)(?=\n *(?:context|conclusion|redacted_text):|$)"
    parts = dict((k.lower(), v.strip()) for k, v in re.findall(pattern, text, flags=re.DOTALL))
    return (
        parts.get("context", ""),
        parts.get("conclusion", ""),
        parts.get("redacted_text", "")
    )

def clean(s):
    return s.replace("\n", " ").replace("\r", " ").strip()


def generate_samples(idx):

    index = idx

    # pre-processing of the {context, question, answer} to {context}
    prompt = " ".join(
        [
            "You are an assistant that rewrites questions into factual statements using the context and answer provided.",
            "Rewrite the context so that it includes the answer in a factual statement.",
            "The output should be a paragraph, not a question and do not add or remove any information.\n",
            f"{df.iloc[index]['question']} {df.iloc[index]['answer']}",
        ]
    )

    response = model.generate_content(f"{prompt}")

    # processing {context} to {incomplete context, conclusion, missing information}
    prompt = (
        "Transform the provided text into three parts.\n"
        "Follow these instructions exactly:\n\n"

        "1. Create a conclusion that is fully supported by the complete context.\n"
        "2. Remove exactly one meaningful line from the context so that the conclusion becomes incomplete.\n"
        "3. Output only the following three fields in order:\n"
        "   context:\n"
        "   conclusion:\n"
        "   redacted_text:\n\n"

        "Formatting rules:\n"
        "- Use the exact labels: context:, conclusion:, redacted_text:\n"
        "- Each label must be followed by its content on the same line.\n"
        "- Do not include any additional text, explanation, or markdown.\n\n"

        "Begin using the text below:\n"
        f"{response.text}"
    )

    resp_abduction = model.generate_content(f"{prompt}").text

    context, conclusion, missing = extract_parts(resp_abduction)


    ## Sanity check to see if extraction was successful
    # print(abductive_dict)
    # print(f"My dict keys: {abductive_dict.keys()}")

    ## NOTE: The keys to the abductive_dict are ['context', 'conclusion', 'redacted_text']
    prompt = "".join(
        [
            "Modified Knowledge (K')\n",
            f"{context}\n",
            "Conclusion (C)\n",
            f"{conclusion}\n",
            "Your Task\n",
            "Based on the Modified Knowledge (K') and the Conclusion (C), ",
            "what is the missing piece of information that would have originally",
            "been placed in the Knowledge to fully support and explain the conclusion?",
            "Please directly give the answer without any explanation.",
        ]
    )

    # Prompt Model
    chat_completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"{prompt}"}],
    )

    # get llama's response
    response1 = chat_completion.choices[0].message.content

    # get gemini's response
    response2 = model.generate_content(f"{prompt}").text

    return map(clean, (context, conclusion, missing, response1, response2))



# create file once + write header
data_path = "data/results.csv"
if not os.path.isfile(data_path):
    with open(data_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Incomplete Knowledge", "conclusion", "Redacted Text", "response1", "response2"])

num_samples = min(10, len(df))

for sample in range(8, num_samples):
    while True:
        try:
            context, conclusion, miss_info, response1, response2 = generate_samples(sample)
            break  # success → exit retry loop
        except Exception as e:
            if "quota" in str(e).lower():
                print("Rate limited. Sleeping 35s...")
                time.sleep(35)
            else:
                raise

    with open(data_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow([context, conclusion, miss_info, response1, response2])

    print(f"Saved row {sample+1}/{num_samples}")

