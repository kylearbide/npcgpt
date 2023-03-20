from bs4 import BeautifulSoup
import os
import json

BASE_PATH = "../../data/knowledge_base/"

def parse_html():
    files = os.listdir(BASE_PATH)
    item_dict = {}
    for file in files:
        with open(BASE_PATH + file, "r") as f:
            data = f.read()

        soup = BeautifulSoup(data)
        items = soup.find_all('a')
        key = file.split(".",1)[0]
        item_dict.update({key:[item.string for item in items]})

    with open(BASE_PATH + "kb.json", "w") as f:
        f.write(json.dumps(item_dict))

if __name__ == "__main__":
    parse_html()
