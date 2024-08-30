from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct
import os
import uuid
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

qdrant_key = os.getenv('QDRANT_API_KEY')
openai_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=openai_key)

connection = QdrantClient(
    url="https://8faa4521-4da6-4819-9e80-57e9b210527f.europe-west3-0.gcp.cloud.qdrant.io:6333/",
    api_key=qdrant_key,
)
# print("Connection successful:", connection)  # for checking connection is successful or not
try:
    connection.create_collection(
        collection_name="pdf-chatbot",
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
    )
    info = connection.get_collection(collection_name="pdf-chatbot")
    print(info)
except Exception as e:
    print("Collection already exists")


def read_data_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_embedding(text_chunks, model_id="text-embedding-ada-002"):
    points = []
    for idx, chunk in enumerate(text_chunks):
        response = client.embeddings.create(
            input=chunk,
            model=model_id
        )
        embeddings = response.data[0].embedding
        point_id = str(uuid.uuid4())
        points.append(PointStruct(id=point_id, vector=embeddings, payload={"text": chunk}))

    return points

def insert_data(get_points):
    operation_info = connection.upsert(
        collection_name="pdf-chatbot",
        wait=True,
        points=get_points
    )
    
    
def main():
    # Initial setup: read PDF and insert data (commented out after first run)
    pdf_path = str(input("Enter PDF path you want to ask from: "))
    get_raw_text = read_data_from_pdf(pdf_path)
    chunks = get_text_chunks(get_raw_text)
    vectors = get_embedding(chunks)
    insert_data(vectors)
    print("data inserted")
    
if __name__ == '__main__':
    main()