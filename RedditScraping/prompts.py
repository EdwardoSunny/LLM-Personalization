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
    emotional state: The person's current mood or emotional state.

In your output, refer to the person simply as "the person." Do not mention Reddit or the original post. Output only the extracted or inferred trait, with no extra text. MAKE SURE you always guess the logical trait if it is not given. You can never say not given, you must guess based off of context.

Here is the paragraph: {input_text}

{format_instructions}
"""

filter_template = """
You are an expert at information extraction. Your task is to determine whether specific information is present in a given paragraph. For each field below, assess whether the text explicitly or implicitly contains the corresponding information. Respond with **true** only if you are confident the information is present; otherwise, respond with **false**.

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

