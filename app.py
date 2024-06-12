from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from pinecone import Pinecone, Index
import os
from langchain.vectorstores import Pinecone as BasePineconeVectorStore
from src.prompt import prompt_template  # Ensure prompt_template is correctly imported

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_HOST = "https://medicalchatbot-n4tyhi0.svc.aped-4627-b74a.pinecone.io"  # Replace with your actual Pinecone host URL

embeddings = download_hugging_face_embeddings()

# Initializing the Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicalchatbot"

# Loading the index
index = pc.Index(index_name)

text_key = "text"  # Specify the metadata field that stores the raw text

class CustomPineconeVectorStore(BasePineconeVectorStore):
    def similarity_search_with_score(self, query, **search_kwargs):
        # Convert the query to an embedding
        query_embedding = embeddings.embed_query(query)
        docs_and_scores = self._index.query(
            vector=query_embedding,
            top_k=search_kwargs['k'],
            include_metadata=True
        )
        return docs_and_scores

docsearch = CustomPineconeVectorStore(
    index=index,
    embedding_function=embeddings.embed_query,
    text_key=text_key
)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}

llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens': 512,
                            'temperature': 0.8})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input_query = msg
    print("Input query: ", input_query)
    try:
        result = qa({"query": input_query})
        print("Full Response: ", result)  # Log the full response to see its structure
        # Ensure 'result' key exists in the response
        if "result" in result:
            return jsonify({"response": result["result"]})
        else:
            print("Unexpected response format:", result)
            return jsonify({"response": "An error occurred while processing your request."})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"response": "An error occurred while processing your request."})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)