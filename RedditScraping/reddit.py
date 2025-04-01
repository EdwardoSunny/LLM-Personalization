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
    
