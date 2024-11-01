import json
import os
from PyPDF2 import PdfReader
from google.oauth2 import service_account
import google.ai.generativelanguage as glm
from google_labs_html_chunker.html_chunker import HtmlChunker

#set up the Service account credentials
service_account_file_name = 'service_account_key.json'
credentials = service_account.Credentials.from_service_account_file(service_account_file_name)
scoped_credentials = credentials.with_scopes(
    ['https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/generative-language.retriever'])

# initialize the clients for google
generative_service_client = glm.GenerativeServiceClient(credentials=scoped_credentials)
retriever_service_client = glm.RetrieverServiceClient(credentials=scoped_credentials)

# function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to split text into smaller chunks respecting token limit. You might want to do this better lol
def split_text_into_chunks(text, max_tokens=1000):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(' '.join(current_chunk + [word])) > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
        current_chunk.append(word)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

#function to chunk text
def chunk_text(text, max_words_per_chunk=200):
    chunker = HtmlChunker(
        max_words_per_aggregate_passage=max_words_per_chunk,
        greedily_aggregate_sibling_nodes=True,
        html_tags_to_exclude={"noscript", "script", "style"}
    )
    passages = chunker.chunk(text)
    return passages

#Function to create a corpus
def create_corpus(display_name):
    example_corpus = glm.Corpus(display_name=display_name)
    create_corpus_request = glm.CreateCorpusRequest(corpus=example_corpus)
    create_corpus_response = retriever_service_client.create_corpus(create_corpus_request)
    return create_corpus_response.name

# get the corpus by display name (you need this)
def get_corpus_by_display_name(display_name):
    request = glm.ListCorporaRequest()
    response = retriever_service_client.list_corpora(request)
    for corpus in response.corpora:
        if corpus.display_name == display_name:
            return corpus.name
    return None

# this function is for creating the document in the corpus
def create_document(corpus_resource_name, display_name, metadata):
    example_document = glm.Document(display_name=display_name)
    example_document.custom_metadata.extend([glm.CustomMetadata(key=k, string_value=v) for k, v in metadata.items()])
    create_document_request = glm.CreateDocumentRequest(parent=corpus_resource_name, document=example_document)
    create_document_response = retriever_service_client.create_document(create_document_request)
    return create_document_response.name

# function to create chunks in a document
def create_chunks(document_resource_name, passages):
    chunks = []
    for passage in passages:
        smaller_chunks = split_text_into_chunks(passage)
        for chunk in smaller_chunks:
            chunks.append(glm.Chunk(data={'string_value': chunk}))

    create_chunk_requests = [glm.CreateChunkRequest(parent=document_resource_name, chunk=chunk) for chunk in chunks]
    request = glm.BatchCreateChunksRequest(parent=document_resource_name, requests=create_chunk_requests)
    response = retriever_service_client.batch_create_chunks(request)
    return response.chunks

# the main function to process PDFs and generate embeddings
def process_pdfs_and_generate_embeddings(pdf_paths, output_json_path, corpus_display_name="My Corpus"):
    #get or create the corpus
    corpus_resource_name = get_corpus_by_display_name(corpus_display_name)
    if not corpus_resource_name:
        corpus_resource_name = create_corpus(corpus_display_name)
    
    embeddings = []

    for pdf_path in pdf_paths:
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_path)
        print(f"Extracted text from {pdf_path}")

        # Chunk the text
        passages = chunk_text(text)
        print(f"Chunked text into {len(passages)} passages")

        # start by making a document in the corpus (10k docs max)
        document_resource_name = create_document(corpus_resource_name, os.path.basename(pdf_path), {"source": pdf_path})
        print(f"Created document {document_resource_name}")

        # make the chunks in the document (so it can be retreieved as chunks)
        created_chunks = create_chunks(document_resource_name, passages)
        print(f"Created {len(created_chunks)} chunks")

        # Collect embeddings for chunks
        for chunk in created_chunks:
            embeddings.append({
                "chunk_id": chunk.name,
                "text": chunk.data.string_value  #this is how you should access string_value
            })
    
    # I'm saving the embeddings to a JSON file so you can see what's going on
    # Note that this is not actually needed in any way, it's all in the cloud
    with open(output_json_path, 'w') as f:
        json.dump(embeddings, f, indent=4)
    print(f"Embeddings saved to {output_json_path}")

if __name__ == "__main__":
    data_folder = "appp/static/data"
    pdf_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.pdf')]
    output_json = "embeddings.json"
    process_pdfs_and_generate_embeddings(pdf_files, output_json)
