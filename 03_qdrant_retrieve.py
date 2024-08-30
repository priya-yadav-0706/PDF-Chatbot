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

def create_answer_with_context(query, history):
    # Include the conversation history in the context
    conversation_history = "\n".join(history)
    prompt = f"{conversation_history}\nUser: {query}\nAI:"
    
    response = client.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    )
    embeddings = response.data[0].embedding
    search_result = connection.search(
        collection_name="pdf-chatbot",
        query_vector=embeddings,
        limit=5
    )
    
    prompt_text = ""
    for result in search_result:
        prompt_text += result.payload['text']
    
    concatenated_string = f"{prompt_text} {query}"
    
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": concatenated_string}
        ],
        stream=True
    )
    
    response_text = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            response_text += chunk.choices[0].delta.content
            # print(chunk.choices[0].delta.content, end='')
    # print()
    
    return response_text.strip()

def main():
    # Initial setup: read PDF and insert data (commented out after first run)
    # get_raw_text = read_data_from_pdf()
    # chunks = get_text_chunks(get_raw_text)
    # vectors = get_embedding(chunks)
    # insert_data(vectors)

    # Start a conversation loop
    conversation_history = []
    print("Start chatting with the bot (type 'exit' to end):")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        # Append user input to conversation history
        conversation_history.append(f"User: {user_input}")
        
        # Generate response
        response = create_answer_with_context(user_input, conversation_history)
        
        # Append AI response to conversation history
        conversation_history.append(f"AI: {response}")
        
        # Print AI response
        print(f"\nAI: {response}\n")

if __name__ == '__main__':
    main()
