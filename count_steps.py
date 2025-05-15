import pandas as pd
import os

models = ["llama3-8b-instruct", "qwen25-7b-instruct", "mistral-7b-instruct", "deepseek-7b", "llama31-8b-instruct", "qwq-32b"]


print("SYTH===============================")
for model in models:
    df = pd.read_csv(os.path.join("agents_output", f"{model}_results.csv"))

    # print the average the "Length" column of the dataframe

    average_length = df["Path Length"].mean()
    print(f"Average length of path for {model}: {average_length}")

print("===============================")


print("REAL===============================")
for model in models:
    categories = ["career", "education", "financial", "health", "life", "relationship", "social"]

    curr_avg = 0
    for category in categories:
        df = pd.read_csv(os.path.join("agents_output", "real", category, f"real_{model}_results.csv"))
        average_length = df["Path Length"].mean()
        curr_avg += average_length

    final_average_length = curr_avg / len(categories)
    print(f"Average length of path for {model}: {final_average_length}")
print("===============================")
