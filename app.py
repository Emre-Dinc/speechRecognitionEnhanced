from flask import Flask, request, send_file, make_response
import os

app = Flask(__name__)


@app.route('/video')
def get_video():
    # Set the path to your video file
    video_path = '"C:/xampp/htdocs/pythonProject/ağaç.mp4"'
    # Check if the file exists
    if os.path.exists(video_path):
        # Send the file to the client
        return send_file(video_path, mimetype='video/mp4')
    else:
        # Return a 404 error if the file doesn't exist
        return 'Video not found', 404


if __name__ == 'main':
    app.run(debug=True)
