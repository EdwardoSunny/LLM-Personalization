import os
import base64
import csv
import re
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
from openai import OpenAI

client = OpenAI()

eval_category = "life"

# MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
# OUTPUT_FILE = f"output/{eval_category}/llama3-8b-instruct_results.csv"

# MODEL = "Qwen/Qwen2.5-7B-Instruct"
# OUTPUT_FILE = f"output/{eval_category}/qwen25-7b-instruct_results.csv"

# MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
# OUTPUT_FILE = f"output/{eval_category}/mistral-7b-instruct_results.csv"

# MODEL = "deepseek-ai/deepseek-llm-7b-chat"
# OUTPUT_FILE = f"output/{eval_category}/deepseek-7b_results.csv"

# Create a vLLM instance using your open source model.
llm = LLM(model=MODEL)

# Define sampling parameters for generating responses.
sampling_params = SamplingParams(max_tokens=1024, temperature=0.7, top_p=0.95)

# Read JSON file
def read_json_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    return None

# contains queries and backgrounds 
real_reddit_data = read_json_file(f"RedditScraping/data/{eval_category}/posts.json")

# Open CSV file for writing results.
output_file = OUTPUT_FILE
print("================")
print("RUNNING")
print("================")
with open(output_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Scenario", "Background", "User Query",
        "Response Without Background", 
        "Response With Background" 
    ])

    # cap at 1000 posts, because some data has 2000 posts
    total_iterations = 1000 

    # Initialize progress bar.
    with tqdm(total=total_iterations, desc="Processing Backgrounds and Queries", unit="query") as pbar:
        for i, profile in enumerate(real_reddit_data):
            # Format the background description.
            background_text = f"""
            - age: {profile['age']}
            - gender: {profile['gender']}
            - marital status: {profile['marital status']}
            - profession: {profile['profession']}
            - economic status: {profile['economic status']}
            - health status: {profile['health status']}
            - education level: {profile['education level']}
            - mental health status: {profile['mental health status']}
            - emotional state: {profile['emotional state']}
            """

            query = profile['query']

            print("Processing background:\n", background_text)
            print("Processing query:\n", query)

            # Prepare prompt without background.
            no_background_prompt = (
                f"User query: {query}\n"
                f"Assistant response:"
            )
            no_background_messages = [
                {"role": "system", "content": "You are an AI assistant that helps people find information."},
                {"role": "user", "content": no_background_prompt}
            ]
            # Prepare prompt with background.
            with_background_prompt = (
                f"User background:\n{background_text}\n"
                f"User query: {query}\n"
                f"Consider the user’s background and identify what the user truly needs based on the provided query\n"
                f"Assistant response:"
            )
            with_background_messages = [
                {"role": "system", "content": "You are an AI assistant that helps people find information."},
                {"role": "user", "content": with_background_prompt}
            ]
            try:
                no_background_outputs = llm.chat(no_background_messages, sampling_params=sampling_params)
                no_background_result = no_background_outputs[0].outputs[0].text
            except Exception as e:
                no_background_result = f"Error: {e}"
                no_background_avg_score = "N/A"

            try:
                with_background_outputs = llm.chat(with_background_messages, sampling_params=sampling_params)
                with_background_result = with_background_outputs[0].outputs[0].text
            except Exception as e:
                with_background_result = f"Error: {e}"
                with_background_avg_score = "N/A"

            # Write results to CSV.
            try:
                writer.writerow([
                    f"User Post {i + 1}",
                    background_text,
                    query,
                    no_background_result, 
                    with_background_result, 
                ])

                file.flush()
            except Exception as e:
                print(f"Error while writing to CSV: {e}")
            pbar.update(1)
print(f"Responses and evaluations saved to {output_file}")
