from bs4 import BeautifulSoup
import requests

r = requests.get(
    "https://drive.google.com/file/d/1iHhZqexvnyX-2MUeQoJcoUIWWJ_3F0LZ/view"
)

result = BeautifulSoup(r.text, "html.parser")

print(result)
