from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

#schema
class Review(TypedDict):
    key_themes: Annotated[list[str], "must write down all the important concepts discussed in the review in a list."]
    summary: Annotated[str, "Must write down a brief summary of the review."]
    sentiment: Annotated[str, "Must write a sentiment of the review--either positive, negative or mixed."]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside of a list."]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside of a list."]

structured_model = model.with_structured_output(Review)

prompt = '''
Xiaomi 15T impresses as a premium-midrange smartphone with a 6.83-inch AMOLED 1.5K display (120 Hz, up to 3200 nits, HDR10+ / Dolby Vision) that delivers rich, bright visuals and smooth animation. Powered by MediaTek Dimensity 8400 Ultra with 12 GB RAM and up to 512 GB storage, it handles everyday tasks, multitasking, and gaming with solid performance and fluid responsiveness. The triple-camera system co-developed with Leica (50 MP main, 50 MP telephoto, 12 MP ultra-wide) produces excellent photos with good detail, natural colours and decent telephoto reach, making it a reliable all-round shooter. Battery life is generous thanks to the 5500 mAh cell and 67 W fast charging, offering dependable endurance for a day and more. On the downside, the phone lacks wireless charging, and while the telephoto lens gives modest 2× zoom, it falls short of pro-level 5× optical zoom found in higher-end models — so camera enthusiasts seeking extreme zoom may find it limiting.'''

result = structured_model.invoke(prompt)

print(result)
