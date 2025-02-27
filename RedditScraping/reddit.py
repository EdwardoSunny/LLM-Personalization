import os
import praw
import pandas as pd
from typing import Annotated, List
from functools import wraps
from datetime import datetime, timezone
from datetime import date, timedelta, datetime


SavePathType = Annotated[str, "File path to save data. If None, data is not saved."]


# def process_output(data: pd.DataFrame, tag: str, verbose: VerboseType = True, save_path: SavePathType = None) -> None:
#     if verbose:
#         print(data.to_string())
#     if save_path:
#         data.to_csv(save_path)
#         print(f"{tag} saved to {save_path}")


def save_output(data: pd.DataFrame, tag: str, save_path: SavePathType = None) -> None:
    if save_path:
        data.to_csv(save_path)
        print(f"{tag} saved to {save_path}")


def get_current_date():
    return date.today().strftime("%Y-%m-%d")


def decorate_all_methods(decorator):
    def class_decorator(cls):
        for attr_name, attr_value in cls.__dict__.items():
            if callable(attr_value):
                setattr(cls, attr_name, decorator(attr_value))
        return cls

    return class_decorator


def init_reddit_client(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global reddit_client
        if not all([os.environ.get("REDDIT_CLIENT_ID"), os.environ.get("REDDIT_CLIENT_SECRET")]):
            print("Please set the environment variables for Reddit API credentials.")
            return None
        else:
            reddit_client = praw.Reddit(
                client_id=os.environ["REDDIT_CLIENT_ID"],
                client_secret=os.environ["REDDIT_CLIENT_SECRET"],
                user_agent="python:subredditSearch:v0.1 (by /u/yourusername)",
            )
            print("Reddit client initialized")
            return func(*args, **kwargs)
    return wrapper


@decorate_all_methods(init_reddit_client)
class RedditUtils:

    def get_subreddit_posts(
        self,
        subreddit: Annotated[str, "The subreddit name to search in"],
        limit: Annotated[int, "Maximum number of posts to fetch, default is 100"] = 100,
        selected_columns: Annotated[
            List[str],
            "Columns to include in the result. Available columns: 'created_utc', 'id', 'title', 'selftext', 'score', 'num_comments', 'url'."
        ] = ["created_utc", "id", "title", "selftext", "score", "num_comments", "url"],
        save_path: SavePathType = None,
    ) -> pd.DataFrame:
        """
        Get the most recent posts from the specified subreddit.
        """
        post_data = []

        subreddit_obj = reddit_client.subreddit(subreddit)
        for post in subreddit_obj.new(limit=limit):
            post_data.append(
                [
                    datetime.fromtimestamp(post.created_utc, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    post.id,
                    post.title,
                    post.selftext,
                    post.score,
                    post.num_comments,
                    post.url,
                ]
            )

        output = pd.DataFrame(
            post_data,
            columns=["created_utc", "id", "title", "selftext", "score", "num_comments", "url"],
        )
        output = output[selected_columns]

        save_output(output, f"reddit posts from subreddit {subreddit}", save_path=save_path)

        return output

if __name__ == "__main__":
    # Ensure you have set the environment variables:
    # REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET before running this script.
    
    # Create an instance of RedditUtils
    reddit_utils = RedditUtils()
    
    # Specify the subreddit and number of posts to fetch
    subreddit_name = "python"  # You can change this to any subreddit you want
    num_posts = 10  # Number of most recent posts to fetch
    
    # Fetch the posts
    posts_df = reddit_utils.get_subreddit_posts(subreddit=subreddit_name, limit=num_posts)
    
    # Display the fetched posts
    print(posts_df.head())
