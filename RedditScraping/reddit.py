import requests
import time
import json
from datetime import datetime, timedelta

class RedditDataFetcher:
    """
    A class to fetch all Reddit submissions between a specified date range
    using the PullPush API, storing only title and content.
    """
    
    def __init__(self, base_url="https://api.pullpush.io/reddit/search"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def fetch_all_submissions_in_date_range(self, start_date, end_date, subreddit=None, 
                                           query=None, output_file="reddit_submissions.json"):
        """
        Fetch all submissions between start_date and end_date, one day at a time.
        
        Args:
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            subreddit (str, optional): Restrict to a specific subreddit
            query (str, optional): Search term
            output_file (str): File to save results
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        all_results = []
        current_date = start
        
        
        while current_date <= end:
            print(f"\nProcessing day: {current_date.strftime('%Y-%m-%d')}")
            next_date = current_date + timedelta(days=1)
            
            # Convert to epoch timestamps
            after_timestamp = int(current_date.timestamp())
            before_timestamp = int(next_date.timestamp())
            
            # Fetch all submissions for this day
            day_results = self.fetch_all_submissions_for_timeframe(
                after_timestamp, 
                before_timestamp,
                subreddit=subreddit,
                query=query
            )
            
            all_results.extend(day_results)
            
            # Move to next day
            current_date = next_date
            
            # Small pause to be nice to the API
            time.sleep(0.5)
        
        # Save results to file
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=4)
        
        return all_results
    
    def fetch_all_submissions_for_timeframe(self, after_timestamp, before_timestamp, 
                                          subreddit=None, query=None):
        """
        Fetch all submissions for a specific timeframe, handling pagination.
        
        Args:
            after_timestamp (int): Start timestamp (epoch)
            before_timestamp (int): End timestamp (epoch)
            subreddit (str, optional): Restrict to a specific subreddit
            query (str, optional): Search term
        
        Returns:
            list: All submissions for the timeframe, containing only title and content
        """
        all_results = []
        batch_size = 100  # Maximum allowed by the API
        total_fetched = 0
        
        while True:
            # Build the request parameters
            params = {
                'after': after_timestamp,
                'before': before_timestamp,
                'size': batch_size,
                'sort': 'asc',  # Get oldest first for consistent pagination
            }
            
            # Add optional parameters if provided
            if subreddit:
                params['subreddit'] = subreddit
            if query:
                params['q'] = query
            
            # Construct URL - specifically for submissions only
            url = f"{self.base_url}/submission/"
            
            # Make the request
            response = self.session.get(url, params=params)
            
            if response.status_code != 200:
                print(f"Error: API returned status code {response.status_code}")
                print(response.text)
                print(params)
                break
            print("Success: API returned data")
            # Parse the response
            try:
                data = response.json()
            except json.JSONDecodeError:
                print("Error: Could not parse API response as JSON")
                break
            
            # Check if we have results
            if not data.get('data') or len(data['data']) == 0:
                break
            
            # Extract only title and content from each submission
            results = data['data']
            filtered_results = []
            
            for post in results:
                filtered_post = {
                    'title': post.get('title', ''),
                    'content': post.get('selftext', ''),
                    'id': post.get('id', ''),
                    'url': post.get('url', ''),
                    'subreddit': post.get('subreddit', ''),
                    'created_utc': post.get('created_utc', 0)
                }
                filtered_results.append(filtered_post)
            
            all_results.extend(filtered_results)
            total_fetched += len(results)
            
            # Print progress 
            print(f"Fetched {total_fetched} submissions so far for this timeframe {after_timestamp} {before_timestamp}...", end='\r')

            # If we got fewer results than the batch size, we've reached the end
            if len(results) < batch_size:
                break
            
            # Otherwise, update the after timestamp to get the next batch
            # Use the created_utc of the last item plus 1 second as the new after
            after_timestamp = results[-1]['created_utc'] + 1
            
            # Small pause to be nice to the API
            time.sleep(0.2)
        
        print(f"Completed fetching {total_fetched} submissions for this timeframe.")
        return all_results


# Example usage
if __name__ == "__main__":
    fetcher = RedditDataFetcher()
    
    # Example: Fetch all submissions from r/AskScience in January 2023
    posts = fetcher.fetch_all_submissions_in_date_range(
        start_date="2025-03-10",
        end_date="2025-03-11",
        subreddit="relationships",
        output_file="relationships.json"
    )
