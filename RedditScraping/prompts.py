extract_query_prompt = """Extract the explicit question from a Reddit post written by someone in crisis. Ensure that the extracted question is self-contained—clear and understandable as if it were directly asked to an AI assistant—while omitting most of the background details. Do not add or infer any information beyond what is explicitly stated in the post.

Here are some examples of how to format it:
- I'm feeling overwhelmed at work. How can I manage my stress better?
- I am in a fight with my boyfriend, he just left the house. How can I manage this situation?
- I just got fired. I want to switch careers but feel scared. What should I consider?
"""

extract_background_prompt = "Given the content of a Reddit post written by an individual in crisis, your task is to extract only the background information and context of the person that led them to this situation. Focus on details such as education, family situation, work history, and any other personal factors that may have contributed to their current crisis. Do not include any information beyond their personal background. Do not make up any information that is not present in the original post."

extract_structured_prompt = """
Your task is to extract a specific trait from a Reddit post, such as gender, marital status, age, etc. If the trait is clearly stated, extract it exactly as written (e.g., "male", "female", "married", "single", "divorced"). If the trait is not directly mentioned, infer it based on the context of the post. For example, if someone mentions worrying about their kids, they are likely over 30. Apply reasonable inference when needed, but do not invent details beyond what can be logically deduced.

Always provide a value for the trait — never say "not given" or "not specified". If the trait is clear, extract it directly. If not, make your best guess based on the post's context. Do not include any reasoning or explanations in your output — only the trait itself.

Here are the traits that may be requested and their definitions:

    scenario: The situation the person is in — what is troubling them or what caused them to make the post.
    age: The person's age.
    gender: The person's gender.
    marital status: The person's marital status.
    profession: The person's job.
    economic status: The person's financial situation — income level, wealth, or class (e.g., poor, middle class).
    health status: Whether the person is healthy. If not, list any specific diseases.
    education level: The highest degree of education the person has obtained.
    mental health status: The person's mental health, including any specific conditions or disorders.
    past self harm history: Whether the person has a history of self-harm.
    emotional state: The person's current mood or emotional state.

In your output, refer to the person simply as "the person." Do not mention Reddit or the original post. Output only the extracted or inferred trait, with no extra text. MAKE SURE you always guess the logical trait if it is not given. You can never say not given, you must guess based off of context.
"""
