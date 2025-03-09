from typing import List
from pydantic import BaseModel, Field

# the query and the background of the user from a reddit post
class CoreContent(BaseModel):
    query: str = Field(description="The explicit question from a Reddit post written by someone in crisis")
    background: str = Field(description="Relevant background information and context about the person that led to their situation.")

# background attributes to extract/infer from a reddit post 
class BackgroundAttributes(BaseModel):
    scenario: str = Field(description="The scenario or situation the person is facing")
    age: str = Field(description="The person's age")
    gender: str = Field(description="The person's gender")
    marital_status: str = Field(description="The person's marital status")
    profession: str = Field(description="The person's profession or occupation")
    economic_status: str = Field(description="The person's economic status")
    health_status: str = Field(description="The person's physical health status")
    education_level: str = Field(description="The person's education level")
    mental_health_status: str = Field(description="The person's mental health status")
    # past_self_harm_history: str = Field(description="Any history of self-harm")
    emotional_state: str = Field(description="The person's current emotional state")

# if the background attributes are present in a paragraph or not 
class BackgroundAttributesPresence(BaseModel):
    scenario: bool = Field(description="Whether the scenario or situation the person is facing is mentioned")
    age: bool = Field(description="Whether the person's age is mentioned")
    gender: bool = Field(description="Whether the person's gender is mentioned")
    marital_status: bool = Field(description="Whether the person's marital status is mentioned")
    profession: bool = Field(description="Whether the person's profession or occupation is mentioned")
    economic_status: bool = Field(description="Whether the person's economic status is mentioned")
    health_status: bool = Field(description="Whether the person's physical health status is mentioned")
    education_level: bool = Field(description="Whether the person's education level is mentioned")
    mental_health_status: bool = Field(description="Whether the person's mental health status is mentioned")
    # past_self_harm_history: bool = Field(description="Whether any history of self-harm is mentioned")
    emotional_state: bool = Field(description="Whether the person's current emotional state is mentioned")
