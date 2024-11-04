# llm-messenger-history

This project creates a personalized chatbot using your Facebook Messenger chat history.

The chatbot can answer questions about your past conversations.

## Features

- Uses your actual chat history to generate responses
- Maintains context across multiple turns in the conversation
- All data processing happens locally, and only relevant context is sent to the OpenAI API

## Prerequisites

- **Python 3.7 or higher**
- An **OpenAI API key**. Get yours from [OpenAI's website](https://platform.openai.com/account/api-keys).

## Setup Instructions

```
python llm_messenger_history.py
```

- Start a Conversation: After running the script, type your questions to interact with the chatbot.
- Exit: Type exit or quit to end the session.

## Example Interaction

```
Chat with your Facebook Messenger AI (type 'exit' to quit):
You: What did I talk about with John last Christmas?
AI: Last Christmas, you and John discussed holiday plans, exchanged gift ideas, and talked about meeting up with friends.
You: Who else was involved in the conversation?
AI: You also mentioned inviting Sarah and Mike to join the holiday gathering.
You: exit
Exiting chat.
```

## Customizations

    •	Adjust Temperature: Modify the temperature parameter in facebook_chatbot.py to control the randomness of the AI’s responses.

`llm = OpenAI(temperature=0.5, openai_api_key=OPENAI_API_KEY)`

    •	Change Chunk Size: Adjust chunk_size and chunk_overlap in the text splitter to optimize performance.

`text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)`
