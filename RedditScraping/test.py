import os
from openai import AzureOpenAI

endpoint = os.getenv("ENDPOINT_URL", "https://oai-b-westus3.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
subscription_key = os.getenv(
    "AZURE_OPENAI_API_KEY",
)
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2024-05-01-preview",
)

response = client.chat.completions.create(
    model=deployment,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you today?"}
    ],
    temperature=0.7
)

# Print the response
print(response.choices[0].message.content)
