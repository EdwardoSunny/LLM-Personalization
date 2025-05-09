import os
import yaml
import json
import pandas as pd
from reddit import RedditDataFetcher  # assuming this module is available
from openai import OpenAI
from prompts import core_content_template, attribute_template
import tqdm
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from models import CoreContent, BackgroundAttributes
from filters import has_all_attributes
import sys
import datetime
import time
import random
import torch
import requests
import cysimdjson
from utils import timeout


class RedditScraper:
    def __init__(self, model="gpt-4o"):
        self.seen_posts = {}

        # model = AzureChatOpenAI(
        #     # azure_endpoint="https://oai-b-westus3.openai.azure.com/",
        #     azure_endpoint="https://kaijie-openai-west-us-3.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-15-preview",
        #     azure_deployment=model,
        #     openai_api_version="2024-05-01-preview",
        #     temperature=0
        # )

        model = ChatOpenAI(model=model, temperature=0)

        self.core_content_parser = JsonOutputParser(pydantic_object=CoreContent)
        self.core_content_prompt = PromptTemplate(
            template=core_content_template,
            input_variables=["input_text"],
            partial_variables={
                "format_instructions": self.core_content_parser.get_format_instructions()
            },
        )
        self.content_chain = self.core_content_prompt | model | self.core_content_parser

        self.attribute_parser = JsonOutputParser(pydantic_object=BackgroundAttributes)
        self.attribute_prompt = PromptTemplate(
            template=attribute_template,
            input_variables=["input_text"],
            partial_variables={
                "format_instructions": self.attribute_parser.get_format_instructions()
            },
        )
        self.attribute_chain = self.attribute_prompt | model | self.attribute_parser

    @timeout(60)
    def format_content(self, original_post):
        try:
            result = self.content_chain.invoke({"input_text": original_post})
            return result["query"], result["background"]
        except Exception as e:
            print(f"{e}...Retrying")
            # sleep randomly between 1 and 5 seconds
            time.sleep(random.uniform(4, 7))
            return self.format_content(original_post)

    @timeout(60)
    def extract_structured_content(self, original_post):
        try:
            result = self.attribute_chain.invoke({"input_text": original_post})
            # check that the result has all the traits. Will throw error if it is not present.
            result["scenario"]
            result["age"]
            result["gender"]
            result["marital_status"]
            result["profession"]
            result["economic_status"]
            result["health_status"]
            result["education_level"]
            result["mental_health_status"]
            result["emotional_state"]
            return result
        except Exception as e:
            print(f"{e}...Retrying")
            # sleep randomly between 1 and 5 seconds
            time.sleep(random.uniform(4, 7))
            return self.extract_structured_content(original_post)

    def run(self, crisis_scenario, total_posts):
        # Load the YAML configuration
        with open("config.yaml", "r") as file:
            config_data = yaml.safe_load(file)
        config = config_data["crisis_scenarios"]

        # Validate that the crisis scenario exists
        if crisis_scenario not in config:
            raise ValueError(
                f"Crisis scenario '{crisis_scenario}' not found in configuration."
            )

        # Get the list of subreddits for the given crisis scenario
        subreddits = config[crisis_scenario]
        num_subreddits = len(subreddits)
        if num_subreddits == 0:
            raise ValueError("No subreddits found for the given scenario.")

        # Determine the number of posts per subreddit (distributing any remainder)
        reddit_utils = RedditDataFetcher()
        today = datetime.datetime.now()
        target_per_subreddit = int(total_posts / len(subreddits))

        # Create the output directory if it doesn't exist
        output_dir = os.path.join("data", crisis_scenario)
        os.makedirs(output_dir, exist_ok=True)
        valid_posts_file = os.path.join(output_dir, "valid_posts.json")
        all_valid_posts = []

        # jsonl parser
        parser = cysimdjson.JSONParser()

        if os.path.exists(valid_posts_file):
            with open(valid_posts_file, "r") as f:
                valid_posts_all = json.load(f)
            # Filter out subreddits that are not in the valid_posts_all
            for sr in valid_posts_all:
                for post in valid_posts_all[sr]:
                    self.seen_posts[post["id"]] = ""
        else:
            valid_posts_all = {}

        print("SEEN POSTS: ", len(self.seen_posts))
        for subreddit in subreddits:
            print(f"\nProcessing subreddit: {subreddit}")
            subreddit_data_path = os.path.join(
                "reddit_data", f"r_{subreddit}_posts.jsonl"
            )
            if not os.path.exists(subreddit_data_path):
                raise Exception(
                    f"Subreddit File Reading Error: {subreddit_data_path} does not exist! Make sure you download the reddit data for this subreddit first!"
                )

            # Check if the subreddit has already been processed
            if subreddit in valid_posts_all:
                curr_subreddit_valid_posts = valid_posts_all[subreddit]
            else:
                curr_subreddit_valid_posts = []

            # If we have enough posts, skip this subreddit
            if len(curr_subreddit_valid_posts) >= target_per_subreddit:
                print(f"Already have enough posts for {subreddit}. Skipping...")
                all_valid_posts = all_valid_posts + curr_subreddit_valid_posts
                continue

            print(
                "Currently for this subreddit, has: ", len(curr_subreddit_valid_posts)
            )

            with open(subreddit_data_path, "rb") as f:
                for i, line in tqdm.tqdm(enumerate(f)):
                    # skip empty lines
                    if not line.strip():
                        continue

                    parsed_line = dict(parser.parse(line))
                    post_content = (
                        f"# {parsed_line['title']}\n\n{parsed_line['selftext']}"
                    )

                    # skip removed posts
                    if "[removed]" in post_content:
                        print("REMOVED POST. Continuing...")
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
                        "created_utc": parsed_line["created_utc"],
                    }

                    if has_all_attributes(post_content):
                        curr_subreddit_valid_posts.append(post)

                    valid_posts_all[subreddit] = curr_subreddit_valid_posts

                    with open(valid_posts_file, "w") as f:
                        json.dump(valid_posts_all, f, indent=4)

                    print(f"Found {len(curr_subreddit_valid_posts)} valid posts so far")
                    if len(curr_subreddit_valid_posts) >= target_per_subreddit:
                        break

            # If we collected too many posts, trim to the target number
            if len(curr_subreddit_valid_posts) > target_per_subreddit:
                curr_subreddit_valid_posts = curr_subreddit_valid_posts[
                    :target_per_subreddit
                ]

            print(
                f"Finished processing {subreddit}: collected {len(curr_subreddit_valid_posts)} posts"
            )
            all_valid_posts = all_valid_posts + curr_subreddit_valid_posts

        output_file = os.path.join(output_dir, "posts.json")
        print("DATA: ", len(all_valid_posts))
        final_data = []

        # After collection is done, switch to ChatOpenAI to avoid content filters
        model = ChatOpenAI(temperature=0.9, model="gpt-4o")
        self.content_chain = self.core_content_prompt | model | self.core_content_parser
        self.attribute_chain = self.attribute_prompt | model | self.attribute_parser

        # Iterate over each row in the original DataFrame.
        for post in tqdm.tqdm(all_valid_posts):
            # Check if the post is already in final_data
            if post["id"] in [p["id"] for p in final_data]:
                print(f"Post {post['id']} already exists in final_data. Skipping...")
                continue

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
                "id": post["id"],
                "title": post["title"],
                "original": post["content"],
                "query": query,
                "background": background,
                # categorization begin
                "scenario": scenario,
                "age": age,
                "gender": gender,
                "marital status": marital_status,
                "profession": profession,
                "economic status": economic_status,
                "health status": health_status,
                "education level": education_level,
                "mental health status": mental_health_status,
                "emotional state": emotional_state,
                "url": post["url"],
                "subreddit": post["subreddit"],
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
