from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

# chat_template
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a very helpful customer support agent."), # 1. Wrapped in tuple ()
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])

# load chat history
chat_history = []

# 2. Added 'r' before the path string to fix SyntaxWarning
with open(r"D:/langchain 27th Nov/chatbot_history.txt") as file:
    chat_history.extend(file.readlines())

prompt = chat_template.invoke({
    "chat_history": chat_history,
    "query": "Where is my refund?"
})

print(prompt)

response = llm.invoke(prompt)

print(response.content)