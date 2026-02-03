from openai import OpenAI
from time import sleep
import json
 
client = OpenAI(
    api_key="up_uEk650WoW92uc784PqUq3l6CbhSqK",
    base_url="https://api.upstage.ai/v2"
)
 
file_path = "document.pdf"  # Replace with your own file
 
with open(file_path, "rb") as f:
    uploaded = client.files.create(
        file=f,
        purpose="user_data"
    )
 
input = [
    {
        "role": "user",
        "content": [
            {
                "type": "input_file",
                "file_id": uploaded.id,
            }
        ],
    }
]
 
resp = client.responses.create(
    model="agt_BxcRatEWzVYH2yRNtyWynn",
    include=["last"],
    input=input,
)
 
while resp.status in {"queued", "in_progress"}:
    sleep(2)
    resp = client.responses.retrieve(
        resp.id,
        include=["last"]
    )
 
if resp.status == "completed":
    print(json.loads(resp.output_text))