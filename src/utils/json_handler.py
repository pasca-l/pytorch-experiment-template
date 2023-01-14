import json


class JsonHandler():
    def __init__(self) -> None:
        pass

    def __call__(self, json_file):
        data = json.load(open(json_file, 'r'))
        return data