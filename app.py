from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings, load_pdf, text_split
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from pinecone import Pinecone, Index
import os
from langchain.vectorstores import Pinecone as BasePineconeVectorStore
from src.prompt import prompt_template, system_prompt

# Initialize Flask application
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Fetch Pinecone API key from environment
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_HOST = "https://medicalchatbot-n4tyhi0.svc.aped-4627-b74a.pinecone.io"

# Download Hugging Face embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalchatbot"
index = pc.Index(index_name)
text_key = "text"  # Metadata field that stores raw text

# Define Document class
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

# Custom Pinecone vector store
class CustomPineconeVectorStore(BasePineconeVectorStore):
    def similarity_search_with_score(self, query, **search_kwargs):
        query_embedding = embeddings.embed_query(query)
        docs_and_scores = self._index.query(
            vector=query_embedding,
            top_k=search_kwargs.get('k', 2),
            include_metadata=True
        )

        if 'matches' in docs_and_scores:
            matches = docs_and_scores['matches']
            if isinstance(matches, list):
                return [
                    (Document(
                        page_content=match['metadata'].get('text', ''),
                        metadata={'score': match['score']}
                    ), match['score'])
                    for match in matches if 'metadata' in match
                ]
            else:
                print("Unexpected 'matches' type:", type(matches))
                return []
        else:
            print("'matches' key not found in response")
            return []

# Create instance of custom Pinecone vector store
docsearch = CustomPineconeVectorStore(
    index=index,
    embedding_function=embeddings.embed_query,
    text_key=text_key
)

# Prompt template for QA
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Initialize CTransformers model
llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens': 256,  # Reduced token limit
                            'temperature': 0.7})    # Adjusted temperature

# Initialize RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# Flask routes
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")
    input_query = msg
    print("Input query:", input_query)

    try:
        # Perform query processing here
        full_query = f"{system_prompt}\n{input_query}"
        result = qa({"query": full_query})
        print("QA Result:", result)
        
        response_text = result['result'] if 'result' in result else "No answer found."
        
        return jsonify({"response": response_text})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"response": "An error occurred while processing your request."})

# Start Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
