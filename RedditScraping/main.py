import os
import yaml
import pandas as pd
from reddit import RedditUtils  # assuming this module is available
from openai import OpenAI
from prompts import extract_background_prompt, extract_query_prompt
import tqdm
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from typing import List

client = OpenAI()

# Define the output schema using Pydantic
class CoreContent(BaseModel):
    query: str = Field(description="The explicit question from a Reddit post written by someone in crisis")
    background: List[str] = Field(description="relevant background information and context about the person that led to their situation.")

# Initialize the output parser
parser = JsonOutputParser(pydantic_object=CoreContent)

# Create a prompt template that includes instructions and the expected output format
template = """
Given a paragraph written by someone in crisis, extract two distinct elements:

1. THE QUESTION: Extract the explicit question as a self-contained query that could be directly asked to an AI assistant. Ensure it's clear and understandable while omitting most background details. Do not add or infer information beyond what is explicitly stated in the post. Format examples:
   - "How can I manage my stress better?"
   - "How can I manage this situation with my boyfriend who just left the house?"
   - "What should I consider when switching careers after being fired?"

2. THE BACKGROUND: Extract only the relevant background information and context about the person that led to their situation. Focus on details such as education, family situation, work history, and personal factors that contributed to their current crisis. Include only information explicitly mentioned in the original post, without making inferences or additions.

Provide these two elements separately and clearly labeled.

Here is the paragraph: {input_text}

{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["input_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Initialize the language model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Create the chain
chain = prompt | model | parser

# Function to analyze text
def format_content(paragraph_text):
    result = chain.invoke({"input_text": paragraph_text})
    print(result["query"])
    print(result["background"])
    return result["query"], result["background"]

# def format_content(original_post):
#     background_input = f"{extract_background_prompt}\n\nHere's the reddit post: {original_post}"
#     query_input = f"{extract_query_prompt}\n\nHere's the reddit post: {original_post}"
    
#     # Get the background information.
#     background_response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "user", "content": background_input},
#         ]
#     )
#     background = background_response.choices[0].message.content

#     # Get the query information.
#     query_response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "user", "content": query_input},
#         ]
#     )
#     query = query_response.choices[0].message.content
#     print("Original Text============")
#     print(original_post)
#     return query, background
  

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
        exit()
        # print("QUERY===========")
        # print(query)
        # print("BACKGROUND===========")
        # print(background)


        # Build a new dictionary with columns in the desired order:
        # 'created_utc', 'id', 'title', 'selftext', 'query', 'background',
        # then the remaining columns.
        new_row = {
            'created_utc': row_dict['created_utc'],
            'id': row_dict['id'],
            'title': row_dict['title'],
            'selftext': row_dict['selftext'],
            'query': query,
            'background': background,
            'score': row_dict['score'],
            'num_comments': row_dict['num_comments'],
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
    # Specify your crisis scenario and total number of posts to fetch across its subreddits
    crisis_scenario = "social"  # change as needed
    total_posts = 200 # total posts you want across all subreddits

    combined_df = fetch_and_save_posts(crisis_scenario, total_posts)
