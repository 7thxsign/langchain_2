from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

# template 1
template1 = PromptTemplate(template="Write a detailed report on {topic}", input_variables=["topic"])


# template 2
template2 = PromptTemplate(template="write a 4-point summary on the following {text}", input_variables=["text"])


parser = StrOutputParser()

# chain
chain = template1 | model | parser | template2 | model | parser

input = {"topic": "Toyota 2KD-FTV and their tunability."}
result = chain.invoke(input)

print(result)