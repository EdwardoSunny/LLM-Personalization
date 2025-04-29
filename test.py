from vllm import LLM, SamplingParams
import torch
import multiprocessing

def extract_final_output(text):
    """
    Parse a text to remove everything before and including the '</think>' tag.
    
    Args:
        text (str): The input text containing a </think> tag
        
    Returns:
        str: The text after the </think> tag, or the original text if no tag is found
    """
    import re
    
    # Look for the </think> tag and extract everything after it
    pattern = r'.*?</think>(.*)'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        # Return the content after the </think> tag
        return match.group(1).strip()
    else:
        # If no </think> tag is found, return the original text
        return text.strip()


def main():
    sampling_params = SamplingParams(max_tokens=1024, temperature=0.7, top_p=0.95)
    llm = LLM(
        model="Qwen/QwQ-32B", 
        dtype=torch.bfloat16, 
        trust_remote_code=True, 
        quantization="bitsandbytes",
        load_format="bitsandbytes"
    )
    # Rest of your code here
    prompt = [
                {"role": "system", "content": "You are an AI assistant that helps people find information."},
                {"role": "user", "content": "Who is the president of the United States? Give me a brief overview of their policies and achievements."}
            ]
    result = llm.chat(prompt, sampling_params=sampling_params)[0].outputs[0].text
    print(result)
    print("===============")
    print(extract_final_output(result))
    
if __name__ == '__main__':
    main()
