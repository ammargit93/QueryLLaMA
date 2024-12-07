
from flask import Flask, request, jsonify, render_template, redirect, url_for
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
import os
import shutil
from pymongo import MongoClient
import atexit

app = Flask(__name__)

client = MongoClient("mongodb://localhost:27017/")
db = client["query_db"]
collection = db["queries"]

folder_path = "db"
os.makedirs(folder_path, exist_ok=True)
os.makedirs("pdf", exist_ok=True)

cached_llm = Ollama(model="llama3")
embedding = FastEmbedEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

raw_prompt = PromptTemplate.from_template(
    """
    <s>[INST] You are a technical assistant good at searching documents. If you do not have an answer from the provided information say so. [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
"""
)

vector_store = None

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload")
def upload():
    return render_template("upload.html")


@app.route("/query/<filename>")
def query_page(filename):
    all_responses = list(collection.find({}, {'_id': 0}))
    return render_template("query.html", filename=filename, responses=all_responses)


@app.route("/ai", methods=["POST"])
def ai_post():
    print("Post /ai called")
    json_content = request.json
    query = json_content.get("query")
    print(f"query: {query}")
    response = cached_llm.invoke(query)
    print(response)
    response_answer = {"answer": response}
    return jsonify(response_answer)


@app.route("/ask_pdf", methods=["POST"])
def ask_pdf_post():
    print("Post /ask_pdf called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    global vector_store
    if vector_store is None:
        print("Loading vector store")
        vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

    print("Creating chain")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 10, "score_threshold": 0.2},
    )

    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    result = chain.invoke({"input": query})
    print(result)

    sources = [{"source": doc.metadata["source"], "page_content": doc.page_content} for doc in result["context"]]
    response_answer = {"answer": result["answer"], "sources": sources}
    print(response_answer['answer'])
    collection.insert_one({"query": query, "answer": response_answer['answer']})

    all_responses = list(collection.find({}, {'_id': 0}))  # Exclude the MongoDB '_id' field from the results

    return render_template('query.html', filename=json_content.get("filename"), responses=all_responses)


@app.route("/pdf", methods=["POST"])
def pdf_post():
    file = request.files["file"]
    file_name = file.filename
    save_file = os.path.join("pdf", file_name)

    clear_chroma_db()

    file.save(save_file)
    print(f"filename: {file_name}")

    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")

    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")

    global vector_store
    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
    )
    vector_store.persist()

    response = {
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks),
    }
    return redirect(url_for('query_page', filename=file_name))


@app.route("/clear_db", methods=["POST"])
def clear_chroma_db():
    global vector_store
    if vector_store is not None:
        vector_store.delete()
        vector_store = None

    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)
    return jsonify({"status": "Database cleared"})


@atexit.register
def cleanup():
    if vector_store is not None:
        vector_store.close()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
