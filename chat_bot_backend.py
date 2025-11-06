#Taking imports
from typing import TypedDict,Literal,Annotated
from pydantic import BaseModel,Field
from langgraph.graph import StateGraph,START,END,add_messages
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage,BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
#defining checkpoint for saving memory at each node:
check_point = MemorySaver()
#loading env file to acess API keys:
load_dotenv()

#Making a state of gaph:
class bot(TypedDict):
  messages : Annotated[list[BaseMessage],add_messages]


#Define LLM model here:
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.1",
)
model = ChatHuggingFace(llm=llm)


#defining thread id for memory purpose:
thread_id = 1
config = {'configurable':{'thread_id':thread_id}}


# Define Function here:
def chat_bot(state:bot):
  mes = state['messages']
  response = model.invoke(mes)
  return {'messages': [response]}


# Define your nodes and edges here:
graph = StateGraph(bot)
graph.add_node('chat_bot',chat_bot)
graph.add_edge(START,'chat_bot')
graph.add_edge('chat_bot',END)
graph_chatbot = graph.compile(checkpointer=check_point)


#logical loop for chatting not needed in streamlit app but required to test in normal python file:

# while True:
#   prompt =  input('user query: ')
#   if prompt == 'exit':
#     break
# response = graph_chatbot.invoke({'messages':HumanMessage(content ="hi my name is moin")},config=config)
#   ai_contents = [msg.content for msg in response["messages"] if isinstance(msg, AIMessage)]
#   print(response['messages'][-1].content)
# messages = graph_chatbot.get_state(config=config)
# print(messages)

graph_chatbot

