import os
import re
import torch
from pprint import pprint
from operator import itemgetter
from bs4 import BeautifulSoup as Soup
from collections import Counter
from flask import request, Flask
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.docstore.document import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

print('-----Loading documents-----')

url = "https://docs.e2enetworks.com/"
loader = RecursiveUrlLoader(
    url=url, max_depth=2, extractor=lambda x: Soup(x, "html.parser").text
)
docs = loader.load()

print('-----Documents loaded------')

def remove_frequent_tokens(text, n_gram_range=(2, 3), min_count=5):
    # Tokenize the text into words and n-grams
    words = re.findall(r'\b\w+\b', text.lower())
    ngrams = []
    for n in range(n_gram_range[0], n_gram_range[1] + 1):
        ngrams.extend(zip(*[words[i:] for i in range(n)]))
    ngrams = [' '.join(ngram) for ngram in ngrams]

    # Count occurrences of words and n-grams
    ngram_counts = Counter(ngrams)
    # print(ngrams)

    # Get the most common words and n-grams
    ngrams_to_remove = {ngram for ngram, count in ngram_counts.items() if count >= min_count and ngram!='e2e networks'}

    # print(ngrams_to_remove)
    # Remove most common words and n-grams
    filtered_text = text
    for token in ngrams_to_remove:
        filtered_text = filtered_text.replace(token, '')

    return filtered_text



def process_docs(doc):
    # replace all newline characeters with a space
    new_doc = doc.replace('\n', ' ')
    #condense multiple space to a single space
    new_doc = re.sub(r'\s+', ' ', new_doc)
    #remove special characters
    new_doc = re.sub(r'[^A-Za-z0-9\s,.\'?-]' ,'',  new_doc)

    # remove frequently occuring n-grams
    new_doc = remove_frequent_tokens(new_doc)
    return new_doc

processed_docs = []
for doc in docs:
    new_doc = process_docs(doc.page_content)
    processed_docs.append(Document(page_content= new_doc, metadata={"source": "local"}))
    # pprint(new_doc)

print('-----Processsing done, now loading model------')

n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="/Users/sagaruprety/Work/Codebases/llama.cpp/models/mistralai_Mistral-7B-Instruct-v0.2/ggml-model-q4_0.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=False,
    n_ctx = 2048 
)


print('-----Model loaded------')

# load the legal case documents

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

docs = text_splitter.split_documents(processed_docs)

print('-----Chunking finished, now loading embeddings------')


# define the embedding model
emb_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=emb_model,
    cache_folder=os.getenv('SENTENCE_TRANSFORMERS_HOME'))


print('-----Embeddings loaded, now ingesting data into Qdrant ------')

qdrant_collection = Qdrant.from_documents(
docs,
embeddings,
location=":memory:", # Local mode with in-memory storage only
collection_name="e2e_documentation",
)
# construct a retriever on top of the vector store
qdrant_retriever = qdrant_collection.as_retriever()

# let's try a query and see the how its retrieved from the qdrant vector database

# define prompt template
template = """<s>[INST] You are a helpful, respectful and honest techinical support assistant for E2E Networks.
Your task is to answer user's questions.
Answer the question below from context below.
{context}
{question} [/INST] </s>
"""

# create the prompt from the above template
prompt = ChatPromptTemplate.from_template(template)

# combine document chunks into one
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# setup the context and question part of the chain
setup_and_retrieval = RunnableParallel(
    {"context": qdrant_retriever| format_docs, "question": RunnablePassthrough()})

# extend the chain to include the prompt, model, and output parser
rag_chain = setup_and_retrieval | prompt | llm | StrOutputParser()

# rag_chain.invoke('What is E2E networks?')

# rag_chain.invoke("How to setup a Kubernetes cluster in E2E networks?")

app = Flask(__name__)

@app.route('/ask_llm', methods=['GET'])
def web_app():
    # Get the input string from the request parameters
    query = request.args.get('query', '')
    llm_response = rag_chain.invoke(query)

    # Return the input string as a text response
    return llm_response, 200, {'Content-Type': 'text/plain'}

if __name__ == '__main__':
    app.run()  

