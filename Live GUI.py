from collections import deque
import datetime as dt
import cv2
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

SEQUENCE_LENGTH=20
LRCN_model=load_model('Models/convlstm_model___Date_Time_2022_08_22__19_23_33___Loss_0.8341655731201172___Accuracy_0.5357142686843872 (2).h5')
CLASSES_LIST = ["Normal","Abnormal"]
video_reader = cv2.VideoCapture(0)
frames_queue = deque(maxlen = SEQUENCE_LENGTH)
LABELS=[0,1]


fig = plt.figure(figsize=(10,3))
ax = fig.add_subplot(1, 1, 1)
xs = []
ys = []

def animate(i, xs, ys):

 ok, frame = video_reader.read()

 frame = cv2.resize(frame, (64, 64))
 normalized_frame = frame / 255
 frames_queue.append(normalized_frame)

 # Check if the number of frames in the queue are equal to the fixed sequence length.
 if len(frames_queue) == SEQUENCE_LENGTH:
  # Pass the normalized frames to the model and get the predicted probabilities.
  predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_queue, axis=0))[0]

  # Get the index of class with highest probability.
  predicted_label = np.argmax(predicted_labels_probabilities)

  # Get the class name using the retrieved index.
  predicted_class_name = CLASSES_LIST[predicted_label]
  print(predicted_class_name)

  frames_queue.clear()

  # Add x and y to lists
  xs.append(dt.datetime.now().strftime('%I:%M:%S'))
  ys.append(predicted_class_name)

 # Limit x and y lists to 20 items
 xs = xs[-20:]
 ys = ys[-20:]

 # Draw x and y lists
 ax.clear()
 ax.plot(xs, ys)

 plt.xticks(rotation=45, ha='right')
 plt.yticks(LABELS, CLASSES_LIST)
 plt.subplots_adjust(bottom=0.30)
 ax.spines.right.set_visible(False)
 ax.spines.top.set_visible(False)

# Set up plot to call animate() function periodically
ani = FuncAnimation(fig, animate, fargs=(xs, ys),frames = 500,
                             interval = 20)

plt.show()
