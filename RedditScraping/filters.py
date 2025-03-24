from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import AzureChatOpenAI
from inspect import get_annotations
from typing import get_type_hints
from models import BackgroundAttributesPresence
from prompts import filter_template

model="gpt-4o"
model = AzureChatOpenAI(
            azure_endpoint="https://oai-b-westus3.openai.azure.com/",
            azure_deployment=model,
            openai_api_version="2024-05-01-preview",
            temperature=0
        )

filter_parser = JsonOutputParser(pydantic_object=BackgroundAttributesPresence)
filter_prompt = PromptTemplate(
    template=filter_template,
    input_variables=["input_text"],
    partial_variables={"format_instructions": filter_parser.get_format_instructions()}
)
filter_chain = filter_prompt | model | filter_parser 

def has_attributes_prescence(original_post):
    try:
        result = filter_chain.invoke({"input_text": original_post})
        return (
            result["scenario"], 
            result["age"], 
            result["gender"], 
            result["marital_status"], 
            result["profession"], 
            result["economic_status"], 
            result["health_status"], 
            result["education_level"], 
            result["mental_health_status"], 
            result["emotional_state"]
        )
    except Exception as e:
        print(f"Error processing text: {str(e)[:100]}...")
        print(f"Orginal Text {original_post}...")
        # Return all False values when parsing fails
        return (False, False, False, False, False, False, False, False, False, False)

def has_all_attributes(original_post):
    if not original_post or len(original_post.strip()) == 0:
        return False
    
    try:
        presences = has_attributes_prescence(original_post)
        
        threshold = 0.7
        
        # Count the number of True values
        attribute_count = sum(1 for presence in presences if presence)
        ratio = attribute_count / len(presences) if presences else 0
        
        print(f"Attribute ratio: {ratio:.2f}")
        print(f"Meets threshold: {ratio >= threshold}")
        
        return ratio >= threshold
    except Exception as e:
        print(f"Error in has_all_attributes: {str(e)}")
        return False
