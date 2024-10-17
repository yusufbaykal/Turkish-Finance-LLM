import pandas as pd
from transformers import pipeline
from datasets import load_dataset

ds = load_dataset("ZixuanKe/sujet-finance-instruct-177k-clean")
data = pd.DataFrame(ds["train"])


def extract_messages(row):
    user_message = row['messages'][0]['content']
    assistant_message = row['messages'][1]['content']
    return pd.Series([user_message, assistant_message])


data[['user_message', 'assistant_message']] = data.apply(extract_messages, axis=1)
flattened_data = data[['user_message', 'assistant_message']]
data_df = flattened_data[["user_message"]]


data_df.to_csv('clean_data.csv', index=False)


translator_pipe_en_to_tr = pipeline("translation_en_to_tr", model="Helsinki-NLP/opus-mt-tc-big-en-tr")

data = pd.read_csv("clean_data.csv")
data['translated_instruction'] = None
max_tokens = 512

translated_samples = []

for i in range(len(data)):
    try:
        row = data.loc[i]
        if isinstance(row['user_message'], str) and row['user_message']:
            original_text = row['user_message'][:max_tokens]
            translated_text = translator_pipe_en_to_tr(original_text)[0]['translation_text']
            data.loc[i, 'translated_instruction'] = translated_text

            if len(translated_samples) < 5:
                translated_samples.append((original_text, translated_text))
        else:
            print(f"Row {i}: geçersiz içerik")
    except Exception as e:
        print(f"Row {i}: çeviri hatası: {e}")
        continue


data.to_csv("translated_finance_data.csv", index=False)
