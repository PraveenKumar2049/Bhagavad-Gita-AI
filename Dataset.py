from datasets import load_dataset
import pandas as pd
dataset = load_dataset("JDhruv14/Bhagavad-Gita_Dataset")
print(dataset)
df = pd.DataFrame(dataset["train"])
print(df.head())
df.to_json("bhagavad_gita_clean.json", orient='records', indent=2,force_ascii=False)
print("Json Exported.")