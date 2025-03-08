from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from typing import List

# Define the output schema using Pydantic
class CoreContent(BaseModel):
    query: str = Field(description="The explicit question from a Reddit post written by someone in crisis")
    background: List[str] = Field(description="relevant background information and context about the person that led to their situation.")

# Initialize the output parser
parser = JsonOutputParser(pydantic_object=CoreContent)

# Create a prompt template that includes instructions and the expected output format
template = """
Given a paragraph written by someone in crisis, extract two distinct elements:

1. THE QUESTION: Extract the explicit question as a self-contained query that could be directly asked to an AI assistant. Ensure it's clear and understandable while omitting most background details. Do not add or infer information beyond what is explicitly stated in the post. Format examples:
   - "How can I manage my stress better?"
   - "How can I manage this situation with my boyfriend who just left the house?"
   - "What should I consider when switching careers after being fired?"

2. THE BACKGROUND: Extract only the relevant background information and context about the person that led to their situation. Focus on details such as education, family situation, work history, and personal factors that contributed to their current crisis. Include only information explicitly mentioned in the original post, without making inferences or additions.

Provide these two elements separately and clearly labeled.

Here is the paragraph: {input_text}

{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["input_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Initialize the language model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Create the chain
chain = prompt | model | parser

# Function to analyze text
def analyze_paragraph(paragraph_text):
    return chain.invoke({"input_text": paragraph_text})

# Example usage
if __name__ == "__main__":
    sample_text = "During the Apollo 11 mission, Neil Armstrong and Buzz Aldrin became the first humans to land on the Moon, while Michael Collins orbited above. The historic event captivated millions around the world and marked a significant achievement in space exploration."
    
    result = analyze_paragraph(sample_text)
    print(f"Topic: {result['topic']}")
    print(f"People mentioned:")
    for person in result['people']:
        print(f"- {person}")
