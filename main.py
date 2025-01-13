from utils import generate_user_profiles, load_config, test_model
import json

config = load_config("config.yaml")
personalized_base_prompt = config["personalized_base_prompt"]
vanilla_base_prompt = config["vanilla_base_prompt"]

num_profiles = 1  # Number of profiles to generate
profiles_big_five = generate_user_profiles(n=num_profiles, use_big_five=True)
profiles_mbti = generate_user_profiles(n=num_profiles, use_big_five=False)

test_model(config["openai_config"], personalized_base_prompt, config["prompt_tasks_dir"], config["prompt_groups"], profiles_big_five, output_dir="output/personalized/")


test_model(config["openai_config"], vanilla_base_prompt, config["prompt_tasks_dir"], config["prompt_groups"], profiles_big_five, output_dir="output/vanilla/")
