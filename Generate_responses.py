import os
import base64
import csv
import re
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
from openai import OpenAI
import torch


# remove reasoning inside of <think> tags from reasoning models outputs like qwq
def extract_final_output(text):
    """
    Parse a text to remove everything before and including the '</think>' tag.

    Args:
        text (str): The input text containing a </think> tag

    Returns:
        str: The text after the </think> tag, or the original text if no tag is found
    """
    import re

    # Look for the </think> tag and extract everything after it
    pattern = r".*?</think>(.*)"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        # Return the content after the </think> tag
        return match.group(1).strip()
    else:
        # If no </think> tag is found, return the original text
        return text.strip()


def main():
    client = OpenAI()

    # MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
    # OUTPUT_FILE = "output/syth_data/llama3-8b-instruct_results.csv"

    # MODEL = "Qwen/Qwen2.5-7B-Instruct"
    # OUTPUT_FILE = "output/syth_data/qwen25-7b-instruct_results.csv"

    # MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
    # OUTPUT_FILE = "output/syth_data/mistral-7b-instruct_results.csv"

    # MODEL = "deepseek-ai/deepseek-llm-7b-chat"
    # OUTPUT_FILE = "output/syth_data/deepseek-7b_results.csv"

    MODEL = "Qwen/QwQ-32B"
    OUTPUT_FILE = f"output/syth_data/qwq-32b_results.csv"

    # Create a vLLM instance using your open source model.
    if "QwQ" in MODEL:
        # For QwQ-32B, use quantization.
        llm = LLM(
            model=MODEL,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            quantization="bitsandbytes",
            load_format="bitsandbytes",
        )
    else:
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

    user_backgrounds = read_json_file("Data/user_profiles_list.json")
    user_queries = read_json_file("Data/queries_list.json")

    # Open CSV file for writing results.
    output_file = OUTPUT_FILE
    print("================")
    print("RUNNING")
    print("================")
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Scenario",
                "Background",
                "User Query",
                "Response Without Background",
                "Response With Background",
            ]
        )

        # Divide queries among backgrounds.
        queries_per_background = len(user_queries) // len(user_backgrounds)
        total_iterations = len(user_queries)

        # Initialize progress bar.
        with tqdm(
            total=total_iterations,
            desc="Processing Backgrounds and Queries",
            unit="query",
        ) as pbar:
            for idx, background in enumerate(user_backgrounds):
                # Get queries for the current background.
                start_index = idx * queries_per_background
                end_index = start_index + queries_per_background
                background_queries = user_queries[start_index:end_index]

                # Process each query.
                for query_idx, query in enumerate(background_queries, start=1):
                    # Format the background description.
                    background_text = "\n".join(
                        [f"- {key}: {value}" for key, value in background.items()]
                    )

                    # Prepare prompt without background.
                    no_background_prompt = (
                        f"User query: {query}\n" f"Assistant response:"
                    )
                    no_background_messages = [
                        {
                            "role": "system",
                            "content": "You are an AI assistant that helps people find information.",
                        },
                        {"role": "user", "content": no_background_prompt},
                    ]
                    # Prepare prompt with background.
                    with_background_prompt = (
                        f"User background:\n{background_text}\n"
                        f"User query: {query}\n"
                        f"Consider the user’s background and identify what the user truly needs based on the provided query\n"
                        f"Assistant response:"
                    )
                    with_background_messages = [
                        {
                            "role": "system",
                            "content": "You are an AI assistant that helps people find information.",
                        },
                        {"role": "user", "content": with_background_prompt},
                    ]
                    try:
                        no_background_outputs = llm.chat(
                            no_background_messages, sampling_params=sampling_params
                        )
                        no_background_result = no_background_outputs[0].outputs[0].text
                    except Exception as e:
                        no_background_result = f"Error: {e}"
                        no_background_avg_score = "N/A"

                    try:
                        with_background_outputs = llm.chat(
                            with_background_messages, sampling_params=sampling_params
                        )
                        with_background_result = (
                            with_background_outputs[0].outputs[0].text
                        )
                    except Exception as e:
                        with_background_result = f"Error: {e}"
                        with_background_avg_score = "N/A"

                    # if is reasoning model, remove the reasoning inside of <think> tags
                    if "QwQ" in MODEL:
                        no_background_result = extract_final_output(
                            no_background_result
                        )
                        with_background_result = extract_final_output(
                            with_background_result
                        )

                    # Print results for debugging.
                    print("================")
                    print(f"Background:\n{background_text}")
                    print(f"Query: {query}")
                    print(f"Response Without Background: {no_background_result}")
                    print(f"Response With Background: {with_background_result}")
                    print("================")

                    # Write results to CSV.
                    try:
                        writer.writerow(
                            [
                                f"Scenario {idx + 1}-{query_idx}",
                                background_text,
                                query,
                                no_background_result,
                                with_background_result,
                            ]
                        )

                        file.flush()
                    except Exception as e:
                        print(f"Error while writing to CSV: {e}")
                    pbar.update(1)
    print(f"Responses and evaluations saved to {output_file}")


if __name__ == "__main__":
    main()
