from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

class Review(BaseModel):
    key_themes: list[str] = Field(description="Write down 2 key themes discussed in the review in a list.")
    summary: str = Field(description="A brief summary of the review.")
    sentiment: Literal["positive", "negative"] = Field(description="Return the sentiment of the review to be either positive or negative.")
    name: Optional[str] = Field(description="Write down the name of the reviewer.")

st_model = model.with_structured_output(Review, strict = True)

prompt = '''
Xiaomi 15T impresses as a premium-midrange smartphone with a 6.83-inch AMOLED 1.5K display (120 Hz, up to 3200 nits, HDR10+ / Dolby Vision) that delivers rich, bright visuals and smooth animation. Powered by MediaTek Dimensity 8400 Ultra with 12 GB RAM and up to 512 GB storage, it handles everyday tasks, multitasking, and gaming with solid performance and fluid responsiveness. The triple-camera system co-developed with Leica (50 MP main, 50 MP telephoto, 12 MP ultra-wide) produces excellent photos with good detail, natural colours and decent telephoto reach, making it a reliable all-round shooter. Battery life is generous thanks to the 5500 mAh cell and 67 W fast charging, offering dependable endurance for a day and more. On the downside, the phone lacks wireless charging, and while the telephoto lens gives modest 2× zoom, it falls short of pro-level 5× optical zoom found in higher-end models — so camera enthusiasts seeking extreme zoom may find it limiting. This review was given by Rande Ravindra'''

response = st_model.invoke(prompt)

print(response)