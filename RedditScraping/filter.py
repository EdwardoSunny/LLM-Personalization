from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from inspect import get_annotations
from typing import get_type_hints
from models import BackgroundAttributesPresence
from prompts import filter_template

model = ChatOpenAI(model="gpt-4o", temperature=0)

filter_parser = JsonOutputParser(pydantic_object=BackgroundAttributesPresence)
filter_prompt = PromptTemplate(
    template=filter_template,
    input_variables=["input_text"],
    partial_variables={"format_instructions": filter_parser.get_format_instructions()}
)
filter_chain = filter_prompt | model | filter_parser 

# # Function to analyze text
def has_attributes_prescence(original_post):
    result = filter_chain.invoke({"input_text": original_post})
    return result["scenario"], result["age"], result["gender"], result["marital_status"], result["profession"], result["economic_status"], result["health_status"], result["education_level"], result["mental_health_status"], result["emotional_state"]

def has_all_attributes(original_post):
    prescences = has_attributes_prescence(original_post)
    
    for i in range(0, len(prescences)):
        if not prescences[i]:
            return False

    return True
