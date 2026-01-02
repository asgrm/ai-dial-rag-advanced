from task._constants import API_KEY
from task.chat.chat_completion_client import DialChatCompletionClient
from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.embeddings.text_processor import TextProcessor, SearchMode
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role

#TODO:
# Create system prompt with info that it is RAG powered assistant.
# Explain user message structure (firstly will be provided RAG context and the user question).
# Provide instructions that LLM should use RAG Context when answer on User Question, will restrict LLM to answer
# questions that are not related microwave usage, not related to context or out of history scope
SYSTEM_PROMPT = """
You are a RAG-powered assistant that assists users with their questions about microwave usage.

## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""

#TODO:
# Provide structured system prompt, with RAG Context and User Question sections.
USER_PROMPT = """##RAG CONTEXT:
{context}


##USER QUESTION:
{query}"""


#TODO:
# - create embeddings client with 'text-embedding-3-small-1' model
embeddings_client = DialEmbeddingsClient(deployment='text-embedding-3-small-1',api_key=API_KEY)

# - create chat completion client
chat_completion_client = DialChatCompletionClient(deployment_name='gpt-4o', api_key=API_KEY)
# - create text processor, DB config: {'host': 'localhost','port': 5433,'database': 'vectordb','user': 'postgres','password': 'postgres'}
# prefered 5436 for local development
db_config = {
             'host': 'localhost',
             'port': 5436,
             'database': 'vectordb',
             'user': 'postgres',
             'password': 'postgres'
}
text_processor = TextProcessor(embeddings_client=embeddings_client, db_config=db_config)


# Create method that will run console chat with such steps:
def main():
    load_context = input("\nLoad context to VectorDB (y/n)? > ").strip().lower()

    if load_context.lower().strip() in ['y', 'yes']:
        text_processor.process_text_file(
            file_name='embeddings/microwave_manual.txt',
            chunk_size=300,
            overlap=50
        )

    conversation = Conversation()
    conversation.add_message(Message(role=Role.SYSTEM, content=SYSTEM_PROMPT))

# - it should run in `while` loop (since it is console chat)
    while True:
# - get user input from console
        query = input("> ").strip()
        if query.lower() == 'exit':
            print("exiting the chat")
            break
# - retrieve context
        context = text_processor.search(
             user_request=query,
             search_mode=SearchMode.EUCLIDIAN_DISTANCE,
             top_k=4,
             score_threshold=0.3
        )

# - perform augmentation
        prompt = USER_PROMPT.format(query=query, context=context)
        conversation.add_message(Message(role=Role.USER, content=prompt))

# - perform generation
        ai_message = chat_completion_client.get_completion(messages=conversation.get_messages(), print_request=True)
        conversation.add_message(Message(role=Role.AI, content=ai_message))

        print(f"RESPONSE:{ai_message.content}")

if __name__ == "__main__":
    main()


# TODO:
#  PAY ATTENTION THAT YOU NEED TO RUN Postgres DB ON THE 5433 WITH PGVECTOR EXTENSION!
#  RUN docker-compose.yml