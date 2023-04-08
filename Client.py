import requests
import av
import os
import numpy as np
from io import BytesIO
import cv2
#Spoken_Words = os.environ['SPOKEN_WORDS']
url = f'https://41d6-88-238-186-243.ngrok-free.app/PythonProject/ağaç.mp4'.format()
response = requests.get(url)
data = response.content

container = av.open(BytesIO(data))
video_stream = next(s for s in container.streams if s.type == 'video')

# Create a window with a name that matches the video title
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Video', video_stream.width, video_stream.height)

# Iterate through the video frames and display them using OpenCV
for packet in container.demux(video_stream):
    for frame in packet.decode():
        image = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_RGB2BGR)
        cv2.imshow('Video', image)
        if cv2.waitKey(1) == ord('q'):
            break

# Cleanup
cv2.destroyAllWindows()