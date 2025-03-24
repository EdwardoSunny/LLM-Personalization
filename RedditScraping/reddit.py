import requests
import time
import json
from datetime import datetime, timedelta
from contextlib import contextmanager


class RedditDataFetcher:
    """
    A class to fetch all Reddit submissions between a specified date range
    using the PullPush API, storing only title and content.
    """
    
    def __init__(self, base_url="https://api.pullpush.io/reddit/search", 
                batch_size=100, request_delay=0.2, day_delay=0.5,
                max_retries=3, retry_delay=5):
        self.base_url = base_url
        self.session = requests.Session()
        self.batch_size = batch_size  # Made configurable
        self.request_delay = request_delay  # Made configurable
        self.day_delay = day_delay  # Made configurable
        self.max_retries = max_retries  # Added for error handling
        self.retry_delay = retry_delay  # Added for error handling
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self):
        """Close the requests session."""
        if hasattr(self, 'session') and self.session:
            self.session.close()
    
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
        # Validate dates
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            
            if start > end:
                raise ValueError("Start date must be before end date")
                
            # Check if dates are in the future
            now = datetime.now()
            if start > now or end > now:
                print("Warning: Date range includes future dates. API may return no results.")
                
        except ValueError as e:
            print(f"Date format error: {str(e)}")
            return []
        
        all_results = []
        current_date = start
        
        while current_date <= end:
            print(f"\nProcessing day: {current_date.strftime('%Y-%m-%d')}")
            next_date = current_date + timedelta(days=1)

            # Convert to epoch timestamps
            after_timestamp = int(current_date.timestamp())
            before_timestamp = int((next_date - timedelta(seconds=1)).timestamp())

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
            time.sleep(self.day_delay)

        # Save results to file
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=4, ensure_ascii=False)
            # print(f"Successfully saved {len(all_results)} submissions to {output_file}")

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
        total_fetched = 0
        
        # Build the base request parameters
        base_params = {
            'after': after_timestamp,
            'before': before_timestamp,
            'size': self.batch_size,
            'sort': 'asc',  # Get oldest first for consistent pagination
        }
        
        # Add optional parameters if provided
        if subreddit:
            base_params['subreddit'] = subreddit
        if query:
            base_params['q'] = query
        
        # Construct URL - specifically for submissions only
        url = f"{self.base_url}/submission/"
        
        last_timestamp = after_timestamp
        same_timestamp_count = 0
        while True:
            # Update the 'after' parameter for each request
            params = base_params.copy()
            params['after'] = last_timestamp
            # Make the request with retry logic
            response = None
            for attempt in range(self.max_retries):
                try:
                    response = self.session.get(url, params=params, timeout=30)
                    if response.status_code == 200:
                        break
                    elif response.status_code == 429:  # Rate limited
                        wait_time = int(response.headers.get('Retry-After', self.retry_delay))
                        # print(f"Rate limited. Waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        # print(f"API error (attempt {attempt+1}/{self.max_retries}): Status {response.status_code}")
                        time.sleep(self.retry_delay)
                except requests.RequestException as e:
                    # print(f"Request failed (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                    time.sleep(self.retry_delay)
            
            if not response or response.status_code != 200:
                # print(f"Failed to fetch data after {self.max_retries} attempts. Moving to next timeframe.")
                break
            
            # Parse the response
            try:
                data = response.json()
            except json.JSONDecodeError:
                # print("Error: Could not parse API response as JSON")
                break
           
            # Check if we have results
            results = data.get('data', [])
            if not results:
                break
            
            
            # Extract only required fields from each submission
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
            
            # Update progress display properly
            # print(f"Fetched {total_fetched} submissions so far for this timeframe...    ", end='\r')
            
            # If we got fewer results than the batch size, we've reached the end
            if len(results) < self.batch_size:
                break
            
            # Get the timestamp of the last item
            new_timestamp = results[-1]['created_utc']
            
            # Check if we're stuck on the same timestamp
            if new_timestamp == last_timestamp:
                same_timestamp_count += 1
                if same_timestamp_count > 100:  # Arbitrary limit to prevent infinite loops
                    print("\nWarning: Multiple posts with identical timestamps. Some posts may be missed.")
                    # Force move ahead by 1 second
                    last_timestamp = new_timestamp + 1
                    same_timestamp_count = 0
                else:
                    # Handle multiple posts with same timestamp by using a unique identifier
                    if len(results) > 0 and 'id' in results[-1]:
                        # Add the ID to the URL to paginate beyond posts with identical timestamps
                        base_params['after_id'] = results[-1]['id']
            else:
                # Different timestamp, update normally and add 1 to avoid missing posts
                last_timestamp = new_timestamp + 1
                same_timestamp_count = 0
                # Remove after_id if it was added
                if 'after_id' in base_params:
                    del base_params['after_id']
            
            # Small pause to be nice to the API
            time.sleep(self.request_delay)
        
        print(f"\nCompleted fetching {total_fetched} submissions for this timeframe.")
        return all_results


# Example usage
if __name__ == "__main__":
    # Using the context manager to automatically close the session
    with RedditDataFetcher() as fetcher:
        # Example: Fetch all submissions from r/relationships in a date range
        posts = fetcher.fetch_all_submissions_in_date_range(
            start_date="2023-01-01",  # Using a past date instead of future
            end_date="2023-01-02",
            subreddit="relationships",
            output_file="relationships.json"
        )
