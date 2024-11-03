import pandas as pd
from transformers import pipeline
from datasets import load_dataset
import csv
import json

translator_pipe_en_to_tr = pipeline("translation_en_to_tr", model="Helsinki-NLP/opus-mt-tc-big-en-tr")

ds = load_dataset("ZixuanKe/sujet-finance-instruct-177k-clean", split="train")

def split_long_message(message, max_length=512):
    return [message[i:i + max_length] for i in range(0, len(message), max_length)]

def translate_messages(messages):
    translated_messages = []
    for message in messages:
        try:
            content_parts = split_long_message(message['content'])
            translated_parts = []

            for part in content_parts:
                translated_content = translator_pipe_en_to_tr(part)[0]['translation_text']
                translated_parts.append(translated_content)
            translated_content = " ".join(translated_parts)

            translated_message = message.copy()
            translated_message['content'] = translated_content
            translated_messages.append(translated_message)
        except Exception as e:
            print(f"Çeviri sırasında hata oluştu: {e}")
            continue
    return translated_messages



with open('translated_finance_data.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['translated_messages'])

    translated_count = 0
    for example in ds:
        messages = example['messages']
        translated_messages = translate_messages(messages)


        writer.writerow([json.dumps(translated_messages, ensure_ascii=False)])

        translated_count += 1
        if translated_count % 500 == 0:
            print(f"{translated_count} satır çevrildi.")
