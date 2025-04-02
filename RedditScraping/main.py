import os
import yaml
import json
import pandas as pd
from reddit import RedditDataFetcher # assuming this module is available
from openai import OpenAI
from prompts import core_content_template, attribute_template
import tqdm
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import AzureChatOpenAI
from models import CoreContent, BackgroundAttributes
from filters import has_all_attributes
import sys
import datetime
import time
import random
import torch
import requests
import cysimdjson

class RedditScraper():
    def __init__(self, model="gpt-4o"):
        self.seen_posts = {}

        model = AzureChatOpenAI(
            azure_endpoint="https://oai-b-westus3.openai.azure.com/",
            azure_deployment=model,
            openai_api_version="2024-05-01-preview",
            temperature=0
        )

        core_content_parser = JsonOutputParser(pydantic_object=CoreContent)
        core_content_prompt = PromptTemplate(
            template=core_content_template,
            input_variables=["input_text"],
            partial_variables={"format_instructions": core_content_parser.get_format_instructions()}
        )
        self.content_chain = core_content_prompt | model | core_content_parser

        attribute_parser = JsonOutputParser(pydantic_object=BackgroundAttributes)
        attribute_prompt = PromptTemplate(
            template=attribute_template,
            input_variables=["input_text"],
            partial_variables={"format_instructions": attribute_parser.get_format_instructions()}
        )
        self.attribute_chain = attribute_prompt | model | attribute_parser 

    def format_content(self, original_post):
        result = self.content_chain.invoke({"input_text": original_post})
        return result["query"], result["background"]

    def extract_structured_content(self, original_post):
        result = self.attribute_chain.invoke({"input_text": original_post})
        return result 

    def run(self, crisis_scenario, total_posts):
        # Load the YAML configuration
        with open('config.yaml', 'r') as file:
            config_data = yaml.safe_load(file)
        config = config_data["crisis_scenarios"]

        # Validate that the crisis scenario exists
        if crisis_scenario not in config:
            raise ValueError(f"Crisis scenario '{crisis_scenario}' not found in configuration.")

        # Get the list of subreddits for the given crisis scenario
        subreddits = config[crisis_scenario]
        num_subreddits = len(subreddits)
        if num_subreddits == 0:
            raise ValueError("No subreddits found for the given scenario.")

        # Determine the number of posts per subreddit (distributing any remainder)
        reddit_utils = RedditDataFetcher()
        all_valid_posts = []
        today = datetime.datetime.now()
        target_per_subreddit = int(total_posts / len(subreddits))

        # Create the output directory if it doesn't exist
        output_dir = os.path.join("data", crisis_scenario)
        os.makedirs(output_dir, exist_ok=True)
        valid_posts_file = os.path.join(output_dir, "valid_posts.json")

        # jsonl parser
        parser = cysimdjson.JSONParser()
       
        for subreddit in subreddits:
            print(f"\nProcessing subreddit: {subreddit}")
            subreddit_data_path = os.path.join("reddit_data", f"r_{subreddit}_posts.jsonl")
            if not os.path.exists(subreddit_data_path):
                raise Exception(f"Subreddit File Reading Error: {subreddit_data_path} does not exist! Make sure you download the reddit data for this subreddit first!")

            curr_subreddit_valid_posts = []

            with open(subreddit_data_path, "rb") as f:
                for i, line in tqdm.tqdm(enumerate(f)):
                    # skip empty lines
                    if not line.strip():
                        continue

                    parsed_line = dict(parser.parse(line))
                    post_content = f"# {parsed_line['title']}\n\n{parsed_line['selftext']}" 

                    # skip removed posts
                    if "[removed]" in post_content:
                        continue

                    # check that we haven't seen this post before
                    if parsed_line["id"] in self.seen_posts:
                        print("HAVE ALREADY SEEN THIS POST")
                        continue
                    else:
                        self.seen_posts[parsed_line["id"]] = ""

                    post = {
                        "title": parsed_line["title"],
                        "content": parsed_line["selftext"],
                        "id": parsed_line["id"],
                        "subreddit": subreddit,
                        "url": parsed_line["url"],
                        "created_utc": parsed_line["created_utc"]
                    }
                    
                    if has_all_attributes(post_content): 
                        curr_subreddit_valid_posts.append(post)

                    print(f"Found {len(curr_subreddit_valid_posts)} valid posts so far")
                    if len(curr_subreddit_valid_posts) >= target_per_subreddit:
                        break

                    with open(valid_posts_file, "w") as f:
                        json.dump(curr_subreddit_valid_posts, f, indent=4)

            # If we collected too many posts, trim to the target number
            if len(curr_subreddit_valid_posts) > target_per_subreddit:
                curr_subreddit_valid_posts = curr_subreddit_valid_posts[:target_per_subreddit]

            print(f"Finished processing {subreddit}: collected {len(curr_subreddit_valid_posts)} posts")
            all_valid_posts = all_valid_posts + curr_subreddit_valid_posts

        output_file = os.path.join(output_dir, "posts.json")
        print("DATA: ", len(all_valid_posts))
        final_data = []
        # Iterate over each row in the original DataFrame.
        for post in tqdm.tqdm(all_valid_posts):
            full_content = post["title"] + ":\n\n" + post["content"]
            # Extract query and background from the selftext column.
            query, background = self.format_content(full_content)
            print("QUERY===========")
            print(query)
            print("BACKGROUND===========")
            print(background)

            print("STRUCTURED===========")
            structured_content_dict = self.extract_structured_content(full_content)
            scenario = structured_content_dict["scenario"]
            age = structured_content_dict["age"]
            gender = structured_content_dict["gender"]
            marital_status = structured_content_dict["marital_status"]
            profession = structured_content_dict["profession"]
            economic_status = structured_content_dict["economic_status"]
            health_status = structured_content_dict["health_status"]
            education_level = structured_content_dict["education_level"]
            mental_health_status = structured_content_dict["mental_health_status"]
            emotional_state = structured_content_dict["emotional_state"]

            print("scenario:", scenario)
            print("age:", age)
            print("gender: ", gender)
            print("marital_status:", marital_status)
            print("profession:", profession)
            print("economic_status:", economic_status)
            print("health_status:", health_status)
            print("education_level:", education_level)
            print("mental_health_status:", mental_health_status)
            print("emotional_state:", emotional_state)

            # Build a new dictionary with columns in the desired order:
            new_post = {
                'id': post['id'],
                'title': post['title'],
                'original': post['content'],
                'query': query,
                'background': background,
                # categorization begin 
                'scenario': scenario,
                'age': age,
                'gender': gender,
                'marital status': marital_status,
                'profession': profession,
                'economic status': economic_status,
                'health status': health_status,
                'education level': education_level,
                'mental health status': mental_health_status,
                'emotional state': emotional_state,
                'url': post['url'],
                'subreddit': post['subreddit']
            }
            final_data.append(new_post)

            # Save the new_rows list as a JSON file
            with open(output_file, "w") as f:
                json.dump(final_data, f, indent=4)

        return final_data 

# Example usage:
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python your_script.py <crisis_scenario> <total_posts>")
        sys.exit(1)

    crisis_scenario = sys.argv[1]
    
    try:
        total_posts = int(sys.argv[2])
    except ValueError:
        print("Error: total_posts must be an integer.")
        sys.exit(1)

    scraper = RedditScraper()
    new_rows = scraper.run(crisis_scenario, total_posts)
