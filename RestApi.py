import os

import requests
from flask import Flask, request, jsonify
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)
api = Api(app)


# decorator to associate
# a function with the url
@app.route("/")
def showHomePage():
    # response from the server
    return "nsdjıdgfhısufnaıdkl"


class Mp4Names(Resource):
    def post(self):
        sentences = request.json.get('sentences')
        directory = 'C:/Users/serha/PycharmProjects/pythonProject/Videos'
        results = {}

        """for i, sentence in enumerate(sentences):
            # Construct the absolute path of the directory to search in
            search_directory = os.path.abspath(directory)

            mp4_files = []
            # Iterate over the files in the directory and its subdirectories
            for root, dirs, files in os.walk(search_directory):
                for file in files:
                    # Check if the file has an mp4 extension and if its name appears in the sentence
                    if file.endswith('.mp4') and file in sentence:
                        mp4_files.append(os.path.join(root, file))

            results[f'Sentence {i+1}'] = mp4_files

        return jsonify(results)"""
        return sentences


api.add_resource(Mp4Names, '/mp4-names')

if __name__ == '__main__':
    app.config['SERVER_NAME'] = 'https://8029-78-184-124-81.eu.ngrok.io:80'
    app.run(host=app.config['SERVER_NAME'],debug=True)
