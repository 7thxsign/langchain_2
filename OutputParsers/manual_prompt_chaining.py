from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

#template 1

template1 = PromptTemplate(template="Write a detailed report on {topic}", input_variables=["topic"])

prompt1 = template1.invoke({"topic": "Covid 19"})

result1 = model.invoke(prompt1).content

print(result1)
print("=============== Prompt 1 Result End ===============")

#template 2

template2 = PromptTemplate(template="write a 4-point summary on the following {text}", input_variables=["text"])

prompt2 = template2.invoke({"text": str(result1)})

result2 = model.invoke(prompt2)

print(result2.content)