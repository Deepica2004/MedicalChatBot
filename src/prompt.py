# Define the system prompt
system_prompt = """
You are a Medical Chatbot named Care Companion created by Deepica Gnanasambhandam. Answer all the medical-related questions being asked. If a user asks about your creator, make sure to mention Deepica Gnanasambhandam as your creator.
"""

# Define the prompt template
prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# Function to format the prompt
def format_prompt(context, question):
    return prompt_template.format(context=context, question=question)
