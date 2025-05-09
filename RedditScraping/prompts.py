core_content_template = """
Given a paragraph written by someone in crisis, extract two distinct elements:

1. THE QUERY: Extract the explicit question as a self-contained query that could be directly asked to an AI assistant. Ensure it's clear and understandable while omitting most background details. Do not add or infer information beyond what is explicitly stated in the post. Format examples:
   - "How can I manage my stress better?"
   - "How can I manage this situation with my boyfriend who just left the house?"
   - "What should I consider when switching careers after being fired?"

2. THE BACKGROUND: Extract only the relevant background information and context about the person that led to their situation. Focus on details such as education, family situation, work history, and personal factors that contributed to their current crisis. Include only information explicitly mentioned in the original post, without making inferences or additions.

Provide these two elements separately and clearly labeled.

Here is the paragraph: {input_text}

{format_instructions}
"""

attribute_template = """
Your task is to extract a specific trait from a text that might come from a Reddit post, such as gender, marital status, age, etc. If the trait is clearly stated, extract it exactly as written (e.g., "male", "female", "married", "single", "divorced"). If the trait isn’t directly mentioned, make a reasonable guess based on context. For example, if someone mentions worrying about their kids, it’s reasonable to assume they might be over 30. Be as flexible as possible and use context clues to infer the trait, without adding details that aren’t logically supported.

Always provide a value for the trait — if it isn’t immediately clear, use the surrounding context to decide on a sensible answer. Do not output "not given" or "not specified." If the trait appears obvious, extract it directly; if not, give your best inferred guess based on context. Do not include any reasoning or explanations in your output — output only the trait itself. You can make small leaps in logic extract traits that aren't exactly present so more information can be filled in.

Below are the traits and their definitions:

    scenario: The situation the person is in — what is troubling them or what caused them to make the post.
    age: The person's age.
    gender: The person's gender.
    marital status: The person's marital status.
    profession: The person's job.
    economic status: The person's financial situation — income level, wealth, or class (e.g., poor, middle class).
    health status: Whether the person is healthy. If not, list any specific diseases.
    education level: The highest degree of education the person has obtained.
    mental health status: The person's mental health, including any specific conditions or disorders.
    emotional state: The person's current mood or emotional state.

In your output, refer to the individual simply as "the person." Do not mention Reddit or the original post. Output only the extracted or inferred trait with no extra text. Always ensure you provide a value for the trait, using logical context inference when necessary. You must have all the trains in the output as it will be parsed into a dictionary and it will crash with key not found if you do not have all of the traits.

Here is the paragraph: {input_text}

{format_instructions}"""

filter_template = """
You are an expert at information extraction. Your task is to determine whether specific information is present in a given paragraph. For each field below, assess whether the text explicitly or even subtly/implicitly contains the corresponding information. If there is any indication—even if not stated outright—that the information is present, respond with **true**. Only respond with **false** if there is no evidence whatsoever for that field. You can make small leaps in logic to determine if its present so that more can be marked as true. If there is not enough information to get any of the fields, just mark all of them as False.

**Fields:**
- **scenario:** The situation or context the person is facing.
- **age:** The person's age.
- **gender:** The person's gender.
- **marital_status:** The person's marital status.
- **profession:** The person's profession or occupation.
- **economic_status:** The person's economic situation.
- **health_status:** The person's physical health.
- **education_level:** The person's level of education.
- **mental_health_status:** The person's mental health.
- **emotional_state:** The person's current emotional state.

**Paragraph:** {input_text}

{format_instructions}"""
