extract_query_prompt = """Extract the explicit question from a Reddit post written by someone in crisis. Ensure that the extracted question is self-contained—clear and understandable as if it were directly asked to an AI assistant—while omitting most of the background details. Do not add or infer any information beyond what is explicitly stated in the post.

Here are some examples of how to format it:
- I'm feeling overwhelmed at work. How can I manage my stress better?
- I am in a fight with my boyfriend, he just left the house. How can I manage this situation?
- I just got fired. I want to switch careers but feel scared. What should I consider?
"""

extract_background_prompt = "Given the content of a Reddit post written by an individual in crisis, your task is to extract only the background information and context of the person that led them to this situation. Focus on details such as education, family situation, work history, and any other personal factors that may have contributed to their current crisis. Do not include any information beyond their personal background. Do not make up any information that is not present in the original post."
