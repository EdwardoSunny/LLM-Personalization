import random
import json
import yaml
from openai import OpenAI
import tqdm

client = OpenAI()
from datetime import datetime
import os

def generate_personality_big_five():
    """
    Returns a dictionary containing randomized values for the Big Five traits
    Each trait is assigned an integer score (1–100). 
    You can adjust the scoring range or distribution to fit your needs.
    """
    return {
        "Openness": random.randint(1, 100),
        "Conscientiousness": random.randint(1, 100),
        "Extraversion": random.randint(1, 100),
        "Agreeableness": random.randint(1, 100),
        "Neuroticism": random.randint(1, 100)
    }

def generate_personality_mbti():
    """
    Returns a random MBTI type from a predefined list. 
    If you prefer MBTI over Big Five, you can call this function in your profiles.
    """
    mbti_types = [
        "ISTJ", "ISFJ", "INFJ", "INTJ",
        "ISTP", "ISFP", "INFP", "INTP",
        "ESTP", "ESFP", "ENFP", "ENTP",
        "ESTJ", "ESFJ", "ENFJ", "ENTJ"
    ]
    return random.choice(mbti_types)

def generate_cultural_background():
    """
    Returns a random cultural background from a predefined list.
    Add or modify entries as needed for your experiment.
    """
    backgrounds = [
        "North American",
        "South American",
        "European",
        "African",
        "Middle Eastern",
        "South Asian",
        "East Asian",
        "Southeast Asian",
        "Oceanian",
        "Caribbean"
    ]
    return random.choice(backgrounds)

def generate_occupation():
    """
    Returns a random occupation from a predefined list.
    Adjust or expand according to your experimental needs.
    """
    occupations = [
        "Healthcare Professional",
        "Educator",
        "Lawyer",
        "Software Engineer",
        "Artist",
        "Financial Analyst",
        "Journalist",
        "Student",
        "Small Business Owner",
        "Researcher"
    ]
    return random.choice(occupations)

def generate_preferences_and_experiences():
    """
    Returns a random subset of 'preferences/experiences' from a predefined list.
    These might indicate emotional sensitivity, privacy concerns, social justice focus, etc.
    """
    possible_experiences = [
        "Emotional sensitivity",
        "Privacy concerns",
        "Social justice advocacy",
        "Chronic illness experience",
        "High stress environment",
        "Frequent traveler",
        "Religious background",
        "LGBTQ+ community involvement",
        "Environmentalist"
    ]
    # Choose 1 to 3 random experiences from the list
    num_choices = random.randint(1, 3)
    return random.sample(possible_experiences, num_choices)

def generate_profile(use_big_five=True):
    """
    Generates a single user profile as a dictionary. 
    If use_big_five is False, it uses MBTI instead of Big Five.
    """
    if use_big_five:
        personality = generate_personality_big_five()
    else:
        personality = {"MBTI": generate_personality_mbti()}

    profile = {
        "Personality": personality,
        "CulturalBackground": generate_cultural_background(),
        "Occupation": generate_occupation(),
        "Preferences_Experiences": generate_preferences_and_experiences()
    }

    return profile

def generate_user_profiles(n=10, use_big_five=True):
    """
    Generates n user profiles (default 10). 
    If use_big_five=False, uses MBTI for personality instead of the Big Five.
    """
    profiles = []
    for _ in range(n):
        profiles.append(generate_profile(use_big_five))
    return profiles

def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
        return config

def test_model(config, base_prompt, task_dir, task_groups, profiles, output_dir=None):
    """
    Calls an OpenAI LLM for each combination of task and user profile.

    :param config: dict containing model details, e.g. {"model_name": "gpt-3.5-turbo", "temperature": 0.7, ...}
    :param base_prompt: str, a base prompt template that will be formatted with {user_profile} and {task_question}
    :param task_dir: str, path to the directory where task JSON files are stored
    :param task_groups: list of str, file names in task_dir (e.g. ["career_questions.json", "salary_questions.json"])
    :param profiles: list of dicts, each dict defining a user's profile
    :param output_dir: directory to write result/logs to

    :return: list of dicts with results grouped by task file:
        [
          {
            "task_file": "career_questions.json",
            "profiles": [
              {
                "profile": {...},
                "responses": [
                  {
                    "task_question": "...",
                    "model_response": "..."
                  },
                  ...
                ]
              },
              ...
            ]
          },
          ...
        ]
    """

    # Make sure you have your OpenAI key set (if using openai library)
    # openai.api_key = "YOUR_API_KEY"

    # Dictionary to hold results keyed by task_file
    task_results = {}

    # Iterate over each task file in the task_groups
    for task_file in tqdm.tqdm(task_groups):
        task_path = os.path.join(task_dir, task_file)

        # Load the list of tasks from the JSON file
        with open(task_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)

        # Create an empty list to hold the (profile -> [responses]) structure for this file
        task_results[task_file] = []

        # For each profile, iterate over every question in the current task file
        for profile in profiles:
            responses_for_this_profile = []

            for task_question in tasks:
                # Convert the profile dict to a JSON-formatted string for clarity
                user_profile_str = json.dumps(profile, indent=2)

                format_values = {
                    "user_profile": user_profile_str if 'user_profile' in base_prompt else '',
                    "task_question": task_question if 'task_question' in base_prompt else ''
                }

                # Format the prompt with only the available values
                final_prompt = base_prompt.format(**format_values)
                print(final_prompt)
                # Call the OpenAI API
                response = client.chat.completions.create(
                    model=config["model"],
                    temperature=config["temperature"],
                    messages=[
                        {"role": "user", "content": final_prompt}
                    ]
                )

                # Extract the model's response
                model_reply = response.choices[0].message.content.strip()

                # Collect each response with its corresponding question
                responses_for_this_profile.append({
                    "task_question": task_question,
                    "model_response": model_reply
                })

            # Once we've gone through all tasks for the given profile, append to the task_file's results
            task_results[task_file].append({
                "profile": profile,
                "responses": responses_for_this_profile
            })

    # Convert our dictionary into a list of dicts so it’s easily serializable:
    # [
    #   {
    #       "task_file": "<filename1>",
    #       "profiles": [
    #           {
    #               "profile": {...},
    #               "responses": [
    #                   {"task_question": "...", "model_response": "..."},
    #                   ...
    #               ]
    #           },
    #           ...
    #       ]
    #   },
    #   ...
    # ]
    organized_results = []
    for task_file, profile_responses in task_results.items():
        organized_results.append({
            "task_file": task_file,
            "profiles": profile_responses
        })

    # If requested, save the results in a timestamped JSON file
    if output_dir is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = config["model"]
        output_file = os.path.join(output_dir, f"{model_name}_results_{timestamp}.json")

        with open(output_file, "w", encoding="utf-8") as outfile:
            json.dump(organized_results, outfile, indent=2)

        print(f"Results saved to {output_file}")

    return organized_results


if __name__ == "__main__":
    # Example usage
    num_profiles = 5  # Number of profiles to generate
    profiles_big_five = generate_user_profiles(n=num_profiles, use_big_five=True)
    profiles_mbti = generate_user_profiles(n=num_profiles, use_big_five=False)

    # Print results in JSON format
    print("=== Big Five Profiles ===")
    print(json.dumps(profiles_big_five, indent=2))

    print("\n=== MBTI Profiles ===")
    print(json.dumps(profiles_mbti, indent=2))
