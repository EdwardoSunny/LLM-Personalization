from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from typing import List

# Define the output schema using Pydantic
class CoreContent(BaseModel):
    query: str = Field(description="The explicit question from a Reddit post written by someone in crisis")
    background: List[str] = Field(description="List of people mentioned in the paragraph")

# Initialize the output parser
parser = JsonOutputParser(pydantic_object=TextAnalysis)

# Create a prompt template that includes instructions and the expected output format
template = """
Analyze the following paragraph and extract the main topic and all people mentioned.

Paragraph: {input_text}

{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["input_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Initialize the language model
model = ChatOpenAI(temperature=0)

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
