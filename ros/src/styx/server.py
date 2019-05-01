#!/usr/bin/env python

import eventlet
eventlet.monkey_patch(socket=True, select=True, time=True)

import eventlet.wsgi
import socketio
import time
from flask import Flask, render_template

from bridge import Bridge
from conf import conf

sio = socketio.Server()
app = Flask(__name__)
msgs = []

dbw_enable = False
imageCount = 0
SKIP_IMAGES = 4
dbw_Count = 0
SKIP_dbw = 10

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)

def send(topic, data):
    s = 1
    msgs.append((topic, data))
    #sio.emit(topic, data=json.dumps(data), skip_sid=True)

bridge = Bridge(conf, send)

@sio.on('telemetry')
def telemetry(sid, data):
    global dbw_enable
    global dbw_Count
    # if data["dbw_enable"] != dbw_enable:    
    dbw_Count += 1
    if dbw_Count >= SKIP_dbw:
        dbw_enable = data["dbw_enable"]
        bridge.publish_dbw_status(dbw_enable)
        dbw_Count = 0
        
    bridge.publish_odometry(data)
    for i in range(len(msgs)):
        topic, data = msgs.pop(0)
        sio.emit(topic, data=data, skip_sid=True)

@sio.on('control')
def control(sid, data):
    bridge.publish_controls(data)

#@sio.on('obstacle')
#def obstacle(sid, data):
#    bridge.publish_obstacles(data)

#@sio.on('lidar')
#def obstacle(sid, data):
#    bridge.publish_lidar(data)

@sio.on('trafficlights')
def trafficlights(sid, data):
    bridge.publish_traffic(data)

@sio.on('image')
def image(sid, data):
    global imageCount 
    imageCount +=1
    if imageCount > SKIP_IMAGES:
        bridge.publish_camera(data)
        imageCount = 0

if __name__ == '__main__':

    # wrap Flask application with engineio's middleware
    if socketio.__version__ == '1.6.1':
        app = socketio.Middleware(sio, app)
    else:
        app = socketio.WSGIApp(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
