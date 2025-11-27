from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {domain} expert."),
    ("human", "Explain in simple terms, the concept of {topic}.")
])

prompt = chat_template.invoke({
    "domain": "Quantum physics",
    "topic": "wormhole"
    }
)

print(prompt)

result = model.invoke(prompt)
print(result.content)