
import os
from openai import AzureOpenAI

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-08-01-preview"
)

# Use the deployment name 'gpt-4' as shown in your endpoint URL
deployment_name = 'gpt-4'

def generate_response(prompt):
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# Test the implementation
if __name__ == "__main__":
    print('Sending a test completion job')
    start_phrase = 'Write a tagline for an ice cream shop.'
    response = generate_response(start_phrase)
    if response:
        print(f"Prompt: {start_phrase}")
        print(f"Response: {response}")
