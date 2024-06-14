from langchain.llms import CTransformers

# Initialize CTransformers model
llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens': 256,  # Reduced token limit
                            'temperature': 0.7})    # Adjusted temperature

# Test the model with a simple query
query = "What is acne?"
try:
    response = llm(query)  # Pass the query directly as a string
    print("Model Response:", response)
except Exception as e:
    print(f"Error: {e}")
