# Turkish Finance Chat: Fine-Tuned Model for Financial Applications

## About the Project

This project aims to develop an AI agent specialized in the finance domain. The **[“Turkish Finance Dataset”](https://huggingface.co/datasets/yusufbaykaloglu/turkish-finance-dataset)** was created by translating and adapting the original dataset into Turkish. The fine-tuning process was performed on the **“ytu-ce-cosmos/Turkish-Llama-8b-DPO-v0.1”** model.

The training process utilized **PEFT (Parameter-Efficient Fine-Tuning)** with the **QLoRA** method, enabling efficient optimization of the model on large datasets. As a result, the model is well-equipped to provide finance-specific knowledge and engage in interactive dialogues.

---

## Technologies Used
- **Model:** ytu-ce-cosmos/Turkish-Llama-8b-DPO-v0.1  
  - **Number of Parameters:** 8 Billion
- **Fine-Tuning:** PEFT and QLoRA methods
- **Dataset:**  
  - Turkish Version: [Turkish Finance Dataset](https://huggingface.co/datasets/yusufbaykaloglu/turkish-finance-dataset)  
  - English Original: [ZixuanKe/sujet-finance-instruct-177k-clean](https://huggingface.co/datasets/ZixuanKe/sujet-finance-instruct-177k-clean)

---

## Model Usage


```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM

config = PeftConfig.from_pretrained("yusufbaykaoglu/turkish-finance-chat")
base_model = AutoModelForCausalLM.from_pretrained("ytu-ce-cosmos/Turkish-Llama-8b-DPO-v0.1")
model = PeftModel.from_pretrained(base_model, "yusufbaykaoglu/turkish-finance-chat")
```

```python
import torch

input_text = "Faiz politikası etkileri nelerdir?"
inputs = tokenizer.encode_plus(input_text, return_tensors="pt", truncation=True)

model.to("cuda")
inputs = {key: value.to("cuda") for key, value in inputs.items()}

with torch.no_grad():
    outputs = model.generate(**inputs, max_length=128)

generated_text = tokenizer.decode(outputs[0])
print(generated_text)
```

```python
# Answer

<|begin_of_text|>Faiz politikası etkileri nelerdir?
Faiz oranı, bir ülkenin para politikalarının en önemli etkilerinden biridir. Bir merkez bankası tarafından faiz oranı belirlendiğinde, ekonomi üzerindeki etkileri derin olabilir.
Faiz oranı, borç verenler ve yatırımcılar için kredi maliyetini belirler.
Faiz oranı yüksek olduğunda, borç almak daha pahalı hale gelir ve kredi talebini azaltır.
Bu, işletmeler ve tüketiciler için borçlanma maliyetlerini artırabilir.
```




