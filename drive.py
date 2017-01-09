import os
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO


import socketIO_client
import tkinter
import threading


# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

orion = socketIO_client.SocketIO('orion', 5678, socketIO_client.LoggingNamespace)

manualSteering = 0.
targetSpeed = 0
throttle = 0.
turnStep = 0.02
steering = 0.

@sio.on('telemetry')
def telemetry(sid, data):
	global manualSteering, throttle, targetSpeed, steering
	# The current steering angle of the car
	steering_angle = data["steering_angle"]
	# The current throttle of the car
	#throttle = data["throttle"]
	# The current speed of the car
	speed = data["speed"]
	throttle = np.clip(.35 * (targetSpeed - float(speed)), -1., 1.)
	# The current image from the center camera of the car
	imgString = data["image"]
	
	if manualSteering == 0:
		# Make call to orion
		orion.emit('predict', imgString, on_predict_response)
	else:
		steering = steering + manualSteering
		orion.emit('update', imgString, steering)
		on_predict_response(steering)

		if abs(manualSteering) > turnStep:
			if manualSteering > 0:
				manualSteering -= turnStep
			else:
				manualSteering += turnStep
		else:
			manualSteering = 0.
	orion.wait_for_callbacks(seconds=1)

def on_predict_response(steering_angle):
	global throttle, steering
	steering = steering_angle
	print("%.3f, %.3f" % (steering_angle, throttle))
	send_control(steering_angle, throttle)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)

def keyPressed(event):
	global manualSteering, targetSpeed, steering
	if event.char == event.keysym:
		print('Normal Key %r' % event.char)
		if event.char == 'q':
			os._exit(0)
	else: # special key
		print('Special Key %r' % event.keysym)
		if event.keysym == 'Left':
			manualSteering -= turnStep
		elif event.keysym == 'Right':
			manualSteering += turnStep
		elif event.keysym == 'Up':
			targetSpeed += 1
		elif event.keysym == 'Down':
			targetSpeed -= 1

		targetSpeed = np.clip(targetSpeed, 0, 30)
		manualSteering = np.clip(manualSteering, -1., 1.)

def send_control(steering_angle, throttle):
    sio.emit("steer", data={'steering_angle': steering_angle.__str__(), 'throttle': throttle.__str__()}, skip_sid=True)

if __name__ == '__main__':
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    def main_loop():
        while True:
            try:
                root.update_idletasks()
                root.update()
            except:
                pass
            eventlet.sleep(0.01)
    root = tkinter.Tk()
    root.bind_all('<Key>', keyPressed)
    eventlet.spawn_after(1, main_loop)


    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
