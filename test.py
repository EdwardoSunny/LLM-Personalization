from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")

print(llm.chat([{"role": "user", "content": "What is the capital of France?"}]))
