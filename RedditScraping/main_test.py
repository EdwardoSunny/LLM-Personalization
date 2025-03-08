import os
import yaml
import pandas as pd
from reddit import RedditUtils  # assuming this module is available
from openai import OpenAI
from prompts import extract_background_prompt, extract_query_prompt, extract_structured_prompt
import tqdm

import sys

client = OpenAI()

def format_content(original_post):
    background_input = f"{extract_background_prompt}\n\nHere's the reddit post: {original_post}"
    query_input = f"{extract_query_prompt}\n\nHere's the reddit post: {original_post}"
    
    # Get the background information.
    background_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": background_input},
        ]
    )
    background = background_response.choices[0].message.content

    # Get the query information.
    query_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": query_input},
        ]
    )
    query = query_response.choices[0].message.content
    print("Original Text============")
    print(original_post)
    return query, background

def extract_structured_content(original_post, trait):
    post_input = f"{extract_structured_prompt}\n\nHere's the reddit post: {original_post}\n\nHere's the trait about the redditer I want you to extract/infer: {trait}"

    background_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": post_input},
        ]
    )
    trait_desc = background_response.choices[0].message.content

    return trait_desc


    
    
  

def fetch_and_save_posts(crisis_scenario, total_posts):
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
    posts_per_subreddit = total_posts // num_subreddits
    remainder = total_posts % num_subreddits

    reddit_utils = RedditUtils()
    posts_dfs = []

    # Fetch posts from each subreddit
    for subreddit in subreddits:
        # Distribute any remainder by adding one extra post to the first few subreddits
        current_limit = posts_per_subreddit + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1
        # Fetch posts as a DataFrame
        df = reddit_utils.get_subreddit_posts(subreddit=subreddit, limit=current_limit)
        # Add a column to identify the subreddit
        df["subreddit"] = subreddit
        posts_dfs.append(df)

    # Combine all fetched posts into one DataFrame
    combined_df = pd.concat(posts_dfs, ignore_index=True)

    # Create the output directory if it doesn't exist
    output_dir = os.path.join("data", crisis_scenario)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "posts.csv")

    # start building new dataframe with the LLM extracted query and background columns
    new_rows = []
    # Iterate over each row in the original DataFrame.
    for index, row in tqdm.tqdm(combined_df.iterrows()):
        # Extract query and background from the selftext column.
        row_dict = row.to_dict()
        query, background = format_content(row_dict["selftext"])
        print("QUERY===========")
        print(query)
        print("BACKGROUND===========")
        print(background)

        print("STRUCTURED===========")
        scenario = extract_structured_content(row_dict["selftext"], "scenario")
        age = extract_structured_content(row_dict["selftext"], "age")
        gender = extract_structured_content(row_dict["selftext"], "gender")
        marital_status = extract_structured_content(row_dict["selftext"], "marital_status")
        profession = extract_structured_content(row_dict["selftext"], "profession")
        economic_status = extract_structured_content(row_dict["selftext"], "economic_status")
        health_status = extract_structured_content(row_dict["selftext"], "health_status")
        education_level = extract_structured_content(row_dict["selftext"], "education_level")
        mental_health_status = extract_structured_content(row_dict["selftext"], "mental_health_status")
        past_self_harm_history = extract_structured_content(row_dict["selftext"], "past_self_harm_history")
        emotional_state = extract_structured_content(row_dict["selftext"], "emotional_state")
        print("scenario:", scenario)
        print("age:", age)
        print("gender: ", gender)
        print("marital_status:", marital_status)
        print("profession:", profession)
        print("economic_status:", economic_status)
        print("health_status:", health_status)
        print("education_level:", education_level)
        print("mental_health_status:", mental_health_status)
        print("past_self_harm_history:", past_self_harm_history)
        print("emotional_state:", emotional_state)
        
        # Build a new dictionary with columns in the desired order:
        # 'created_utc', 'id', 'title', 'selftext', 'query', 'background',
        # then the remaining columns.
        new_row = {
            # 'created_utc': row_dict['created_utc'],
            'id': row_dict['id'],
            'title': row_dict['title'],
            'original': row_dict['selftext'],
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
            'past self harm history': past_self_harm_history,
            'emotional state': emotional_state,
            # 'score': row_dict['score'],
            # 'num_comments': row_dict['num_comments'],
            'url': row_dict['url'],
            'subreddit': row_dict['subreddit']
        }

        new_rows.append(new_row)

    # Create a new DataFrame with the new rows.
    combined_df = pd.DataFrame(new_rows)

    # Save the combined DataFrame as a CSV file
    combined_df.to_csv(output_file, index=False)

    return combined_df


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

    combined_df = fetch_and_save_posts(crisis_scenario, total_posts)
