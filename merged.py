from cscore import CameraServer
import ntcore
from ntcore import NetworkTableInstance
from enum import Enum
import configparser
import robotpy_apriltag
import cv2
import numpy as np
import time
import os
import os.path
import ast
import math
import struct
from math import log10, floor
import json
from picamera2 import Picamera2
import libcamera
from libcamera import controls
import threading
from pprint import *
import sys
import pickle
from cscore import VideoSource, UsbCamera, MjpegServer
from ntcore import EventFlags



X_RES = 320
Y_RES = 240
UPTIME_UPDATE_INTERVAL = 1
TEMP_UPDATE_INTERVAL= 30
DEBUG_MODE_DEFAULT = False
THREADS_DEFAULT = 3
DECIMATE_DEFAULT = 1.0
BLUR_DEFAULT = 0.0
REFINE_EDGES_DEFAULT = True
SHARPENING_DEFAULT = 0.25
APRILTAG_DEBUG_MODE_DEFAULT = False
DECISION_MARGIN_DEFAULT = 125
CAMERA_CAL_FILE_NAME = "MultiMatrix.npz.PiGS.640.480" # "MultiMatrix.npz" #"MultiMatrix.npz.PiGS.320.240" #"MultiMatrix.npz" #MultiMatrix.npz.PiGS.640.480" # "MultiMatrix.npz.PiGS.320.240" # "MultiMatrix.npz.webcam.320.240" # "MultiMatrix.npz.webcam.640.480"
THREADS_TOPIC_NAME = "/Vision/Threads"
DECIMATE_TOPIC_NAME = "/Vision/Decimate"
BLUR_TOPIC_NAME = "/Vision/Blur"
REFINE_EDGES_TOPIC_NAME = "/Vision/Edge Refine"
SHARPENING_TOPIC_NAME = "/Vision/Sharpening"
APRILTAG_DEBUG_MODE_TOPIC_NAME = "/Vision/April Tag Debug"
DECISION_MARGIN_MIN_TOPIC_NAME = "/Vision/Decision Margin Min"
DECISION_MARGIN_MAX_TOPIC_NAME = "/Vision/Decision Margin Max"
TAG_CONFIG_FILE_TOPIC_NAME = "/Vision/Tag Config File"
#ACTIVE_TOPIC_NAME = "/Vision/Active"
TAG_ACTIVE_TOPIC_NAME = "/Vision/Tag Active" 
CORAL_ACTIVE_TOPIC_NAME = "/Vision/Coral Active" 
CAGE_ACTIVE_TOPIC_NAME = "/Vision/Coral Active"
POSE_DATA_RAW_TOPIC_NAME = "Tag Pose Data Bytes" #cannot say /Vision becuase we already do in NTGetRaw
CORAL_POSE_DATA_RAW_TOPIC_NAME = "Coral Pose Data Bytes" #cannot say /Vision becuase we already do in NTGetRaw
CAGE_POSE_DATA_RAW_TOPIC_NAME = "Cage Pose Data Bytes" #cannot say /Vision becuase we already do in NTGetRaw
POSE_DATA_STRING_TOPIC_NAME_HEADER ="/Vision/Pose Data Header"
CORAL_POSE_DATA_STRING_TOPIC_NAME_HEADER = "/Vision/Coral Pose Data Header"
CAGE_POSE_DATA_STRING_TOPIC_NAME_HEADER = "/Vision/Cage Pose Data Header"
POSE_DATA_STRING_TOPIC_NAME_DATA_TRANSLATION ="/Vision/Pose Data Trans"
POSE_DATA_STRING_TOPIC_NAME_DATA_ROTATION ="/Vision/Pose Data Rot"
TAG_PI_TEMP_TOPIC_NAME = "/Vision/Tag Temperature"
CORAL_PI_TEMP_TOPIC_NAME = "/Vision/Coral Temperature"
CAGE_PI_TEMP_TOPIC_NAME = "/Vision/Cage Temperature"
RIO_TIME_TOPIC_NAME = "/Vision/RIO Time"

Z_IN_TOPIC_NAME = "/Vision/Z In"
CORAL_MIN_HUE_TOPIC_NAME = "/Vision/Coral Min Hue"
CORAL_MIN_SAT_TOPIC_NAME = "/Vision/Coral Min Sat"
CORAL_MIN_VAL_TOPIC_NAME = "/Vision/Coral Min Val"
CORAL_MAX_HUE_TOPIC_NAME = "/Vision/Coral Max Hue"
CORAL_MAX_SAT_TOPIC_NAME = "/Vision/Coral Max Sat"
CORAL_MAX_VAL_TOPIC_NAME = "/Vision/Coral Max Val"
CAGE_MIN_HUE_RED_TOPIC_NAME = "/Vision/Cage Min Hue Red"
CAGE_MIN_SAT_RED_TOPIC_NAME = "/Vision/Cage Min Sat Red"
CAGE_MIN_VAL_RED_TOPIC_NAME = "/Vision/Cage Min Val Red"
CAGE_MAX_HUE_RED_TOPIC_NAME = "/Vision/Cage Max Hue Red"
CAGE_MAX_SAT_RED_TOPIC_NAME = "/Vision/Cage Max Sat Red"
CAGE_MAX_VAL_RED_TOPIC_NAME = "/Vision/Cage Max Val Red"
CAGE_MIN_HUE_BLUE_TOPIC_NAME = "/Vision/Cage Min Hue Blue"
CAGE_MIN_SAT_BLUE_TOPIC_NAME = "/Vision/Cage Min Sat Blue"
CAGE_MIN_VAL_BLUE_TOPIC_NAME = "/Vision/Cage Min Val Blue"
CAGE_MAX_HUE_BLUE_TOPIC_NAME = "/Vision/Cage Max Hue Blue"
CAGE_MAX_SAT_BLUE_TOPIC_NAME = "/Vision/Cage Max Sat Blue"
CAGE_MAX_VAL_BLUE_TOPIC_NAME = "/Vision/Cage Max Val Blue"
CORAL_CONFIG_FILE_TOPIC_NAME = "/Vision/Coral Config File"
CORAL_CONFIG_FILE_DEFAULT = "coral_config.ini"
CAGE_CONFIG_FILE_TOPIC_NAME = "/Vision/Cage Config File"
CAGE_CONFIG_FILE_DEFAULT = "cage_config.ini"
TAG_CONFIG_FILE_DEFAULT = "tag_config.ini"
GEN_CONFIG_FILE_DEFAULT = "gen_config.ini"
CORAL_MIN_HUE = 0
CORAL_MIN_SAT = 0
CORAL_MIN_VAL = 0
CORAL_MAX_HUE = 179
CORAL_MAX_SAT = 255
CORAL_MAX_VAL = 255
CAGE_MIN_HUE_RED = 0
CAGE_MIN_SAT_RED = 0
CAGE_MIN_VAL_RED = 0
CAGE_MAX_HUE_RED = 179
CAGE_MAX_SAT_RED = 255
CAGE_MAX_VAL_RED = 255
CAGE_MIN_HUE_BLUE = 0
CAGE_MIN_SAT_BLUE = 0
CAGE_MIN_VAL_BLUE = 0
CAGE_MAX_HUE_BLUE = 179
CAGE_MAX_SAT_BLUE = 255
CAGE_MAX_VAL_BLUE = 255
TAG_ENABLE_TOPIC_NAME = "/Vision/Tag Enable"
CORAL_ENABLE_TOPIC_NAME = "/Vision/Coral Enable"
CAGE_ENABLE_TOPIC_NAME = "/Vision/Cage Enable"
TOP_LINE_DIST_FROM_TOP = 0.15
BOTTOM_LINE_DIST_FROM_TOP = 0.7
CORAL_MIN_AREA_TOPIC_NAME = "/Vision/Coral Min Area"
CORAL_MIN_AREA = 23 #275
CORAL_ANGLE_TOPIC_NAME = "/Vision/Coral Angle"
CAGE_MIN_AREA_TOPIC_NAME = "/Vision/Cage Min Area"
CAGE_MIN_AREA = 600 #275
CAGE_ANGLE_TOPIC_NAME = "/Vision/Cage Angle"
WRITE_TAG_IMAGE = False
TAG_RECORD_ENABLE_TOPIC_NAME = "/Vision/Tag Record"
TAG_RECORD_REMOVE_TOPIC_NAME = "/Vision/Tag Remove"
CORAL_RECORD_DATA_TOPIC_NAME = "/Vision/Coral Record"
CORAL_X_OFFSET = 0
CORAL_Y_OFFSET = 5
CAGE_RECORD_DATA_TOPIC_NAME = "/Vision/Cage Record"
CAGE_X_OFFSET = 0
CAGE_Y_OFFSET = 0 #5 - 16 # Was 5
CORAL_MIN_CENTER_Y = 17
CORAL_MAX_CENTER_Y = 460
CAGE_MIN_CENTER_Y = 17
CAGE_MAX_CENTER_Y = 333
CAGE_CENTER_Y_CLOSE = 260
CAGE_MIN_ASPECT_RATIO = .2
CAGE_MAX_ASPECT_RATIO = .7
CAGE_MIN_EXTENT = .20
CAGE_MAX_EXTENT = .9
CAGE_MIN_EXTENT_CLOSE = .3
CAGE_MAX_EXTENT_CLOSE = .95
CAGE_MIN_EXTENT_PERP = .6
CAGE_MAX_EXTENT_PERP = .73
CAGE_MAX_AR_CLOSE = .8
CORAL_MIN_ASPECT_RATIO = .3
CORAL_MAX_ASPECT_RATIO = 3.5
CAGE_MIN_DISTANCE = 2
CAGE_MAX_DISTANCE = 92
CORAL_MIN_DISTANCE = 0
CORAL_MAX_DISTANCE = 84
CAGE_MIN_ANGLE = -55
CAGE_MAX_ANGLE = 55
CORAL_MIN_ANGLE = 0
CORAL_MAX_ANGLE = 90
Y_CROP = 0
FPS_NUM_SAMPLES = 100 #after this number of images the fps average is calulated
CORAL_Y_CROP_TOPIC_NAME = "/Vision/Coral Y Crop"
CORAL_NUM_PIXELS_FROM_CENTER_BLANK = 15
CAGE_Y_CROP_TOPIC_NAME = "/Vision/Cage Y Crop"
CAGE_NUM_PIXELS_FROM_CENTER_BLANK = 15
TAG_BRIGHTNESS_TOPIC_NAME = "/Vision/Tag Brightness"
CORAL_BRIGHTNESS_TOPIC_NAME = "/Vision/Coral Brightness"
CAGE_BRIGHTNESS_TOPIC_NAME = "/Vision/Cage Brightness"

BRIGHTNESS_DEFAULT = 0.0
TAG_CONTRAST_TOPIC_NAME = "/Vision/Tag Contrast"
CORAL_CONTRAST_TOPIC_NAME = "/Vision/Coral Contrast"
CAGE_CONTRAST_TOPIC_NAME = "/Vision/Cage Contrast"
CONTRAST_DEFAULT = 1.0

CORAL_Y_OFFSET_TOPIC_NAME = "/Vision/Coral Y Offset"
CAGE_Y_OFFSET_TOPIC_NAME = "/Vision/Cage Y Offset"

TAG_ERRORS_TOPIC_NAME = "/Vision/Tag Corrected Errors"
TAG_ERRORS_DEFAULT = 0
TAG_AE_TOPIC_NAME = "/Vision/Tag Auto Exposure"
CORAL_AE_TOPIC_NAME = "/Vision/Coral Auto Exposure"
CAGE_AE_TOPIC_NAME = "/Vision/Cage Auto Exposure"
AE_DEFAULT = True
TAG_EXPOSURE_TOPIC_NAME = "/Vision/Tag Manual Exposure" # only used if AE_TOPIC_NAME is disabled
CORAL_EXPOSURE_TOPIC_NAME = "/Vision/Coral Manual Exposure" # only used if AE_TOPIC_NAME is disabled
CAGE_EXPOSURE_TOPIC_NAME = "/Vision/Cage Manual Exposure" # only used if AE_TOPIC_NAME is disabled
EXPOSURE_DEFAULT = 1000 # in microseconds - total guess as default 
POSE_DATA_X_DEG_TOPIC_NAME = "/Vision/X Deg"
POSE_DATA_Y_DEG_TOPIC_NAME = "/Vision/Y Deg"
POSE_DATA_Z_DEG_TOPIC_NAME = "/Vision/Z Deg"
POSE_DATA_X_IN_TOPIC_NAME = "/Vision/X In"
POSE_DATA_Y_IN_TOPIC_NAME = "/Vision/Y In"
TAG_DETECTED_ID_TOPIC_NAME = "/Vision/Tag Id"
TAG_DETECTED_DM_TOPIC_NAME = "/Vision/Tag DM"
TAG_DETECTED_ERRORS_TOPIC_NAME = "/Vision/Tag Errors"

ALLIANCE_TYPE_TOPIC_NAME = "/FMSInfo/IsRedAlliance"
PERP_TOPIC_NAME = "/Vision/Is Perpenduclar"
USE_CAGE_CAM_STREAM = True
CAGE_DISTANCE_ADJUSTMENT = 2 # inches
CAGE_ANGLE_ADJUSTMENT = 1 # degrees
class NTConnectType(Enum):
    SERVER = 1
    CLIENT = 2
class NTGetString:
    def __init__(self, stringTopic: ntcore.StringTopic, init, default, failsafe):
        self.init = init
        self.default = default
        self.failsafe = failsafe
        # start subscribing; the return value must be retained.
        # the parameter is the default value if no value is available when get() is called
        self.stringTopic = stringTopic.getEntry(failsafe)

        self.stringTopic.setDefault(default)
        self.stringTopic.set(init)

    def get(self):
        return self.stringTopic.get(self.failsafe)

    def set(self, string):
        self.stringTopic.set(string)

    def unpublish(self):
        # you can stop publishing while keeping the subscriber alive
        self.stringTopic.unpublish()

    def close(self):
        # stop subscribing/publishing
        self.stringTopic.close()
class NTGetBoolean:
    def __init__(self, boolTopic: ntcore.BooleanTopic, init, default, failsafe):
        self.init = init
        self.default = default
        self.failsafe = failsafe

        # start subscribing; the return value must be retained.
        # the parameter is the default value if no value is available when get() is called
        self.boolTopic = boolTopic.getEntry(failsafe)

        self.boolTopic.setDefault(default)
        self.boolTopic.set(init)

    def get(self):
        return self.boolTopic.get(self.failsafe)
    def set(self, boolean):
        self.boolTopic.set(boolean)
    def unpublish(self):
        # you can stop publishing while keeping the subscriber alive
        self.boolTopic.unpublish()

    def close(self):
        # stop subscribing/publishing
        self.boolTopic.close()
class NTGetDouble:
    def __init__(self, dblTopic: ntcore.DoubleTopic, init, default, failsafe):
        self.init = init
        self.default = default
        self.failsafe = failsafe
        # start subscribing; the return value must be retained.
        # the parameter is the default value if no value is available when get() is called
        self.dblTopic = dblTopic.getEntry(failsafe)
        self.dblTopic.setDefault(default)
        self.dblTopic.set(init)

    def get(self):
        return self.dblTopic.get(self.failsafe)

    def set(self, double):
        self.dblTopic.set(double)

    def unpublish(self):
        # you can stop publishing while keeping the subscriber alive
        self.dblTopic.unpublish()

    def close(self):
        # stop subscribing/publishing
        self.dblTopic.close()
class NTGetRaw:
    def __init__(self, ntinst, topicname, init, default, failsafe):
        self.init = init
        self.default = default
        self.failsafe = failsafe
        self.table = ntinst.getTable("/Vision")

        self.pub = self.table.getRawTopic(topicname).publish("raw")

    def set(self, raw):
        self.pub.set(raw)

    def unpublish(self):
        # you can stop publishing while keeping the subscriber alive
        self.pub.unpublish()

    def close(self):
        # stop subscribing/publishing
        self.pub.close()
#!/usr/bin/python3

# These two are only needed for the demo code below the FrameServer class.
import time
from threading import Condition, Thread

from picamera2 import Picamera2


class FrameServer:
    def __init__(self, picam2, stream='main'):
        """A simple class that can serve up frames from one of the Picamera2's configured streams to multiple other threads.

        Pass in the Picamera2 object and the name of the stream for which you want
        to serve up frames.
        """
        self._picam2 = picam2
        self._stream = stream
        self._array = None
        self._condition = Condition()
        self._running = True
        self._count = 0
        self._thread = Thread(target=self._thread_func, daemon=True)

    @property
    def count(self):
        """A count of the number of frames received."""
        return self._count

    def start(self):
        """To start the FrameServer, you will also need to start the Picamera2 object."""
        self._thread.start()

    def stop(self):
        """To stop the FrameServer

        First stop any client threads (that might be
        blocked in wait_for_frame), then call this stop method. Don't stop the
        Picamera2 object until the FrameServer has been stopped.
        """
        self._running = False
        self._thread.join()

    def _thread_func(self):
        count = 0
        while self._running:
            array = self._picam2.capture_array(self._stream)
            self._count += 1
            ''' # uncomment this block to see exposure time for images
            count += 1
            if (count > 1000):
                print(self._picam2.capture_metadata()['ExposureTime'])
                count = 0
            '''
            with self._condition:
                self._array = array
                self._condition.notify_all()

    def wait_for_frame(self, previous=None):
        """You may optionally pass in the previous frame that you got last time you called this function.

        This will guarantee that you don't get duplicate frames
        returned in the event of spurious wake-ups, and it may even return more
        quickly in the case where a new frame has already arrived.
        """
        with self._condition:
            if previous is not None and self._array is not previous:
                return self._array
            while True:
                self._condition.wait()
                if self._array is not previous:
                    return self._array


def coral_regress_distance(y):
    terms = [
    -2.5190261681829952e+002,
     2.6573959253152246e+000,
    -7.3688136100950232e-003,
     6.0807308128165267e-006
    ]
    
    t = 1
    r = 0
    for c in terms:
        r += c * t
        t *= y
    return r

def coral_regress_px_per_deg(x):
    terms = [
     3.0950248997394785e+000,
     1.5607142915418259e-001,
    -1.9222516478834060e-003,
     8.9708743706644227e-006
    ]   

    t = 1
    r = 0
    for c in terms:
        r += c * t
        t *= x
    return r

def cage_regress_distance(y):
    
    terms = [
        1.5841590981588232e+002,
        -1.4720781685440105e+000,
        5.5831522903344180e-003,
        -7.7508494000204533e-006
    ]
    
    t = 1
    r = 0
    for c in terms:
        r += c * t
        t *= y
    return r

def cage_regress_px_per_deg(x):
    terms = [
     2.6951913744312228e+000,
     1.6418065324623948e-001,
    -2.1592428271868368e-003,
     1.0464701880711019e-005
    ]   

    t = 1
    r = 0
    for c in terms:
        r += c * t
        t *= x
    return r

def cage_pose_data_string(sequence_num, rio_time, time, dist, angle, perp):
    string_header = f'num={sequence_num} t_rio={rio_time:1.3f} t_img={time:1.3f} z_in={dist:3.1f} y_deg={angle:3.1f} perp={perp}'
    
    return string_header

def coral_pose_data_string(sequence_num, rio_time, time, dist, angle, rot):
    string_header = f'num={sequence_num} t_rio={rio_time:1.3f} t_img={time:1.3f} z_in={dist:3.1f} y_deg={angle:3.1f} rot={rot:3.1f}'

    return string_header

def file_write_corals(file,
                min_h,
                min_s,
                min_v,
                max_h,
                max_s,
                max_v,
                contrast,
                ae,
                exposure,
                y_offset,
                brightness,
                y_crop):
    parser = configparser.ConfigParser()

    parser.add_section('VISION')
    parser.set('VISION', CORAL_CONFIG_FILE_TOPIC_NAME, str(file))
    parser.set('VISION', CORAL_MIN_HUE_TOPIC_NAME, str(round(min_h)))
    parser.set('VISION', CORAL_MIN_SAT_TOPIC_NAME, str(round(min_s)))
    parser.set('VISION', CORAL_MIN_VAL_TOPIC_NAME, str(round(min_v)))
    parser.set('VISION', CORAL_MAX_HUE_TOPIC_NAME, str(round(max_h)))
    parser.set('VISION', CORAL_MAX_SAT_TOPIC_NAME, str(round(max_s)))
    parser.set('VISION', CORAL_MAX_VAL_TOPIC_NAME, str(round(max_v)))
    parser.set('VISION', CORAL_CONTRAST_TOPIC_NAME, str((contrast)))
    parser.set('VISION', CORAL_AE_TOPIC_NAME, str(round(ae)))
    parser.set('VISION', CORAL_EXPOSURE_TOPIC_NAME, str(round(exposure)))
    parser.set('VISION', CORAL_Y_OFFSET_TOPIC_NAME, str(round(y_offset)))
    parser.set('VISION', CORAL_BRIGHTNESS_TOPIC_NAME, str((brightness)))
    parser.set('VISION', CORAL_Y_CROP_TOPIC_NAME, str(round(y_crop)))
    
    with open(file, 'w') as config:
        parser.write(config)
        print('wrote coral file:')
        print({'VISION': dict(parser['VISION'])})

def file_write_cages(file,
                min_h_red,
                min_s_red,
                min_v_red,
                max_h_red,
                max_s_red,
                max_v_red,
                min_h_blue,
                min_s_blue,
                min_v_blue,
                max_h_blue,
                max_s_blue,
                max_v_blue,
                contrast,
                ae,
                exposure,
                y_offset,
                brightness,
                y_crop):

    parser = configparser.ConfigParser()

    print("gloop")

    parser.add_section('VISION')
    parser.set('VISION', CAGE_CONFIG_FILE_TOPIC_NAME, str(file))
    parser.set('VISION', CAGE_MIN_HUE_RED_TOPIC_NAME, str(round(min_h_red)))
    parser.set('VISION', CAGE_MIN_SAT_RED_TOPIC_NAME, str(round(min_s_red)))
    parser.set('VISION', CAGE_MIN_VAL_RED_TOPIC_NAME, str(round(min_v_red)))
    parser.set('VISION', CAGE_MAX_HUE_RED_TOPIC_NAME, str(round(max_h_red)))
    parser.set('VISION', CAGE_MAX_SAT_RED_TOPIC_NAME, str(round(max_s_red)))
    parser.set('VISION', CAGE_MAX_VAL_RED_TOPIC_NAME, str(round(max_v_red)))
    parser.set('VISION', CAGE_MIN_HUE_BLUE_TOPIC_NAME, str(round(min_h_blue)))
    parser.set('VISION', CAGE_MIN_SAT_BLUE_TOPIC_NAME, str(round(min_s_blue)))
    parser.set('VISION', CAGE_MIN_VAL_BLUE_TOPIC_NAME, str(round(min_v_blue)))
    parser.set('VISION', CAGE_MAX_HUE_BLUE_TOPIC_NAME, str(round(max_h_blue)))
    parser.set('VISION', CAGE_MAX_SAT_BLUE_TOPIC_NAME, str(round(max_s_blue)))
    parser.set('VISION', CAGE_MAX_VAL_BLUE_TOPIC_NAME, str(round(max_v_blue)))
    parser.set('VISION', CAGE_CONTRAST_TOPIC_NAME, str((contrast)))
    parser.set('VISION', CAGE_AE_TOPIC_NAME, str(round(ae)))
    parser.set('VISION', CAGE_EXPOSURE_TOPIC_NAME, str(round(exposure)))
    parser.set('VISION', CAGE_Y_OFFSET_TOPIC_NAME, str(round(y_offset)))
    parser.set('VISION', CAGE_BRIGHTNESS_TOPIC_NAME, str((brightness)))
    parser.set('VISION', CAGE_Y_CROP_TOPIC_NAME, str(round(y_crop)))

    with open(file, 'w') as config:
        parser.write(config)
        print('wrote cage file:')
        print({'VISION': dict(parser['VISION'])})

def get_type():
    parser= configparser.ConfigParser()
    parser.read('gen_config.ini')
    return(parser.get('GENERAL', 'type'))

def camera_upside_down():
    parser= configparser.ConfigParser()
    parser.read('gen_config.ini')
    return(parser.getboolean('GENERAL', 'camera_upside_down'))

def file_read_gen(parser, configfile_failure_ntt):
    config_exists = os.path.isfile(GEN_CONFIG_FILE_DEFAULT)
    if config_exists == True:
        parser.read(GEN_CONFIG_FILE_DEFAULT)
        configfile_failure_ntt.set(False) #if it works mark no error
        print('read gen file:')
        print({'GENERAL': dict(parser['GENERAL'])})

    else: # re-create config and container file to default
        configfile_failure_ntt.set(True) # set error for config file

        parser.add_section('GENERAL')
        parser.set('GENERAL', 'Type', 'None')
        parser.set('GENERAL', 'Camera Upside Down', False)
       
        with open("/home/pi/" + GEN_CONFIG_FILE_DEFAULT, 'w') as config:
            parser.write(config)
            print('wrote gen file:')
            print({'GENERAL': dict(parser['GENERAL'])})

        configfile_failure_ntt.set(True) # recreated config file

def file_read_coral(parser, configfile_failure_ntt):
    config_exists = os.path.isfile(CORAL_CONFIG_FILE_DEFAULT)
    if config_exists == True:
        parser.read(CORAL_CONFIG_FILE_DEFAULT)
        configfile_failure_ntt.set(False) #if it works mark no error
        print('read coral file:')
        print({'VISION': dict(parser['VISION'])})
    else: # re-create config and container file to default
        configfile_failure_ntt.set(True) # set error for config file

        parser.add_section('VISION')
        
        parser.set('VISION', CORAL_CONFIG_FILE_TOPIC_NAME, str(CORAL_CONFIG_FILE_DEFAULT))
        parser.set('VISION', CORAL_MIN_HUE_TOPIC_NAME, str(CORAL_MIN_HUE))
        parser.set('VISION', CORAL_MIN_SAT_TOPIC_NAME, str(CORAL_MIN_SAT))
        parser.set('VISION', CORAL_MIN_VAL_TOPIC_NAME, str(CORAL_MIN_VAL))
        parser.set('VISION', CORAL_MAX_HUE_TOPIC_NAME, str(CORAL_MAX_HUE))
        parser.set('VISION', CORAL_MAX_SAT_TOPIC_NAME, str(CORAL_MAX_SAT))
        parser.set('VISION', CORAL_MAX_VAL_TOPIC_NAME, str(CORAL_MAX_VAL))
        parser.set('VISION', 'Brightness', str(BRIGHTNESS_DEFAULT))
        parser.set('VISION', 'Contrast', str(CONTRAST_DEFAULT))
        parser.set('VISION', 'Auto Exposure', str(AE_DEFAULT))
        parser.set('VISION', 'Manual Exposure Time', str(EXPOSURE_DEFAULT))
        parser.set('VISION', 'Y Offset', str(CORAL_Y_OFFSET))
        parser.set('VISION', 'Y Crop', str(Y_CROP))


        with open("/home/pi/" + CORAL_CONFIG_FILE_DEFAULT, 'w') as config:
            parser.write(config)
            print('wrote coral file:')
            print({'VISION': dict(parser['VISION'])})
        configfile_failure_ntt.set(False) # config file recreated

def file_read_cage(parser, configfile_failure_ntt):
    config_exists = os.path.isfile(CAGE_CONFIG_FILE_DEFAULT)
    if config_exists == True:
        parser.read(CAGE_CONFIG_FILE_DEFAULT)
        configfile_failure_ntt.set(False) #if it works mark no error
        print('read cage file:')
        print({'VISION': dict(parser['VISION'])})
    else: # re-create config and container file to default
        configfile_failure_ntt.set(True) # set error for config file

        parser.add_section('VISION')
        
        parser.set('VISION', CAGE_CONFIG_FILE_TOPIC_NAME, str(CAGE_CONFIG_FILE_DEFAULT))
        parser.set('VISION', CAGE_MIN_HUE_RED_TOPIC_NAME, str(CAGE_MIN_HUE_RED))
        parser.set('VISION', CAGE_MIN_SAT_RED_TOPIC_NAME, str(CAGE_MIN_SAT_RED))
        parser.set('VISION', CAGE_MIN_VAL_RED_TOPIC_NAME, str(CAGE_MIN_VAL_RED))
        parser.set('VISION', CAGE_MAX_HUE_RED_TOPIC_NAME, str(CAGE_MAX_HUE_RED))
        parser.set('VISION', CAGE_MAX_SAT_RED_TOPIC_NAME, str(CAGE_MAX_SAT_RED))
        parser.set('VISION', CAGE_MAX_VAL_RED_TOPIC_NAME, str(CAGE_MAX_VAL_RED))
        parser.set('VISION', CAGE_MIN_HUE_BLUE_TOPIC_NAME, str(CAGE_MIN_HUE_BLUE))
        parser.set('VISION', CAGE_MIN_SAT_BLUE_TOPIC_NAME, str(CAGE_MIN_SAT_BLUE))
        parser.set('VISION', CAGE_MIN_VAL_BLUE_TOPIC_NAME, str(CAGE_MIN_VAL_BLUE))
        parser.set('VISION', CAGE_MAX_HUE_BLUE_TOPIC_NAME, str(CAGE_MAX_HUE_BLUE))
        parser.set('VISION', CAGE_MAX_SAT_BLUE_TOPIC_NAME, str(CAGE_MAX_SAT_BLUE))
        parser.set('VISION', CAGE_MAX_VAL_BLUE_TOPIC_NAME, str(CAGE_MAX_VAL_BLUE))
        parser.set('VISION', 'Brightness', str(BRIGHTNESS_DEFAULT))
        parser.set('VISION', 'Contrast', str(CONTRAST_DEFAULT))
        parser.set('VISION', 'Auto Exposure', str(AE_DEFAULT))
        parser.set('VISION', 'Manual Exposure Time', str(EXPOSURE_DEFAULT))
        parser.set('VISION', 'Y Offset', str(CAGE_Y_OFFSET))
        parser.set('VISION', 'Y Crop', str(Y_CROP))

        with open("/home/pi/" + CAGE_CONFIG_FILE_DEFAULT, 'w') as config:
            parser.write(config)
            print('wrote cage file:')
            print({'VISION': dict(parser['VISION'])})
        configfile_failure_ntt.set(False) # config file recreated

def nt_update_corals(config,
              configfile,
              min_h,
              min_s,
              min_v,
              max_h,
              max_s,
              max_v,
              brightness,
              contrast,
              ae,
              exposure,
              y_offset,
              y_crop):
    # sync the stuff in the file with matching values in the file

    print('dump coral file:')
    print({'VISION': dict(config['VISION'])})

    mi_h = float(config.get('VISION', CORAL_MIN_HUE_TOPIC_NAME))
    mi_s = float(config.get('VISION', CORAL_MIN_SAT_TOPIC_NAME))
    mi_v = float(config.get('VISION', CORAL_MIN_VAL_TOPIC_NAME))
    mx_h = float(config.get('VISION', CORAL_MAX_HUE_TOPIC_NAME))
    mx_s = float(config.get('VISION', CORAL_MAX_SAT_TOPIC_NAME))
    mx_v = float(config.get('VISION', CORAL_MAX_VAL_TOPIC_NAME))
    bright = float(config.get('VISION', CORAL_BRIGHTNESS_TOPIC_NAME))
    cont = float(config.get('VISION', CORAL_CONTRAST_TOPIC_NAME))
    a_ex = bool(config.get('VISION', CORAL_AE_TOPIC_NAME))
    ex = float(config.get('VISION', CORAL_EXPOSURE_TOPIC_NAME))
    y_off = float(config.get('VISION', CORAL_Y_OFFSET_TOPIC_NAME))
    y_crp = float(config.get('VISION', CORAL_Y_CROP_TOPIC_NAME))

    #configfile.set(str(config.get('VISION', CORAL_CONFIG_FILE_TOPIC_NAME)))
    min_h.set(mi_h)
    min_s.set(mi_s)
    min_v.set(mi_v)
    max_h.set(mx_h)
    max_s.set(mx_s)
    max_v.set(mx_v)
    brightness.set(bright)
    contrast.set(cont)
    ae.set(a_ex)
    exposure.set(ex)
    y_offset.set(y_off)
    y_crop.set(y_crp)


def nt_update_cages(config,
              configfile,
              min_h_red,
              min_s_red,
              min_v_red,
              max_h_red,
              max_s_red,
              max_v_red,
              min_h_blue,
              min_s_blue,
              min_v_blue,
              max_h_blue,
              max_s_blue,
              max_v_blue,
              brightness,
              contrast,
              ae,
              exposure,
              y_offset,
              y_crop):
    
    # sync the stuff in the file with matching values in the file

    print('dump cage file:')
    print({'VISION': dict(config['VISION'])})

    mi_h_r = float(config.get('VISION', CAGE_MIN_HUE_RED_TOPIC_NAME))
    mi_s_r = float(config.get('VISION', CAGE_MIN_SAT_RED_TOPIC_NAME))
    mi_v_r = float(config.get('VISION', CAGE_MIN_VAL_RED_TOPIC_NAME))

    mx_h_r = float(config.get('VISION', CAGE_MAX_HUE_RED_TOPIC_NAME))
    mx_s_r = float(config.get('VISION', CAGE_MAX_SAT_RED_TOPIC_NAME))
    mx_v_r = float(config.get('VISION', CAGE_MAX_VAL_RED_TOPIC_NAME))

    mi_h_b = float(config.get('VISION', CAGE_MIN_HUE_BLUE_TOPIC_NAME))
    mi_s_b = float(config.get('VISION', CAGE_MIN_SAT_BLUE_TOPIC_NAME))
    mi_v_b = float(config.get('VISION', CAGE_MIN_VAL_BLUE_TOPIC_NAME))

    mx_h_b = float(config.get('VISION', CAGE_MAX_HUE_BLUE_TOPIC_NAME))
    mx_s_b = float(config.get('VISION', CAGE_MAX_SAT_BLUE_TOPIC_NAME))
    mx_v_b = float(config.get('VISION', CAGE_MAX_VAL_BLUE_TOPIC_NAME))

    bright = float(config.get('VISION', CAGE_BRIGHTNESS_TOPIC_NAME))
    cont = float(config.get('VISION', CAGE_CONTRAST_TOPIC_NAME))
    a_ex = bool(config.get('VISION', CAGE_AE_TOPIC_NAME))
    ex = float(config.get('VISION', CAGE_EXPOSURE_TOPIC_NAME))
    y_off = float(config.get('VISION', CAGE_Y_OFFSET_TOPIC_NAME))
    y_crp = float(config.get('VISION', CAGE_Y_CROP_TOPIC_NAME))

    #configfile.set(str(config.get('VISION', CAGE_CONFIG_FILE_TOPIC_NAME)))
    min_h_red.set(mi_h_r)
    min_s_red.set(mi_s_r)
    min_v_red.set(mi_v_r)
    max_h_red.set(mx_h_r)
    max_s_red.set(mx_s_r)
    max_v_red.set(mx_v_r)
    min_h_blue.set(mi_h_b)
    min_s_blue.set(mi_s_b)
    min_v_blue.set(mi_v_b)
    max_h_blue.set(mx_h_b)
    max_s_blue.set(mx_s_b)
    max_v_blue.set(mx_v_b)
    brightness.set(bright)
    contrast.set(cont)
    ae.set(a_ex)
    exposure.set(ex)
    y_offset.set(y_off)
    y_crop.set(y_crp)


'''
all data to send is packaged as an array of bytes, using a Python bytearray, in big-endian format:
sequence number: unsigned long (4 bytes)
rio time: float (4 bytes)
image time:float (4 bytes)
type (tag = 1, coral = 3): unsigned char (1 byte)
length: how many tags/corals follow
what follows these first 3 items depends on the type:
tag:
number of tags detected: unsigned char (1 byte)
for each tag: tag id unsigned char (1 byte), pose x: float (4 bytes), pose y: float (4 bytes), pose z: float (4 bytes), pose x angle: float (4 bytes), pose y angle: float (4 bytes), pose z angle: float (4 bytes)

coral:
number of corals detected: unsigned char (1 byte)
for each coral: pose x: float (4 bytes), pose y: float (4 bytes), pose z: float (4 bytes), pose x angle: float (4 bytes), pose y angle: float (4 bytes), pose z angle: float (4 bytes)s)
'''
def cage_pose_data_bytes(sequence_num, rio_time, image_time, type, dist, angle, perp):
    byte_array = bytearray()
    # start the array with sequence number, the RIO's time, image time, and tag type
    byte_array += struct.pack(">LffBB", sequence_num, rio_time, image_time, type, 1)
    byte_array += struct.pack(">ff?", angle, dist, perp) 
    return byte_array

def coral_pose_data_bytes(sequence_num, rio_time, image_time, type, dist, angle, rot):
    byte_array = bytearray()
    # start the array with sequence number, the RIO's time, image time, and tag type
    byte_array += struct.pack(">LffBB", sequence_num, rio_time, image_time, type, 1)
    byte_array += struct.pack(">fff", angle, dist, rot) 
    return byte_array

def remove_image_files(path):
    for filename in os.listdir(path): 
        file_path = os.path.join(path, filename)  
        if os.path.isfile(file_path):
            os.remove(file_path)  


streamconfigFile = "/boot/frc-stream.json"

class streamCameraConfig: pass

team = None
streamserver = False
streamcameraConfigs = []
streamswitchedCameraConfigs = []
streamcameras = []

def streamparseError(str):
    """Report parse error."""
    print("config error in '" + configFile + "': " + str, file=sys.stderr)

def streamreadCameraConfig(config):
    """Read single camera configuration."""
    cam = streamCameraConfig()

    # name
    try:
        cam.name = config["name"]
    except KeyError:
        parseError("could not read camera name")
        return False

    # path
    try:
        cam.path = config["path"]
    except KeyError:
        parseError("camera '{}': could not read path".format(cam.name))
        return False

    # stream properties
    cam.streamConfig = config.get("stream")

    cam.config = config

    streamcameraConfigs.append(cam)
    return True

def streamreadSwitchedCameraConfig(config):
    """Read single switched camera configuration."""
    cam = streamCameraConfig()

    # name
    try:
        cam.name = config["name"]
    except KeyError:
        parseError("could not read switched camera name")
        return False

    # path
    try:
        cam.key = config["key"]
    except KeyError:
        parseError("switched camera '{}': could not read key".format(cam.name))
        return False

    streamswitchedCameraConfigs.append(cam)
    return True

def streamreadConfig():
    """Read configuration file."""
    global team
    global streamserver

    # parse file
    try:
        with open(streamconfigFile, "rt", encoding="utf-8") as f:
            j = json.load(f)
    except OSError as err:
        print("could not open '{}': {}".format(streamconfigFile, err), file=sys.stderr)
        return False

    # top level must be an object
    if not isinstance(j, dict):
        parseError("must be JSON object")
        return False

    # team number
    try:
        team = j["team"]
    except KeyError:
        parseError("could not read team number")
        return False

    # ntmode (optional)
    if "ntmode" in j:
        str = j["ntmode"]
        if str.lower() == "client":
            streamserver = False
        elif str.lower() == "server":
            streamserver = True
        else:
            parseError("could not understand ntmode value '{}'".format(str))

    # cameras
    try:
        streamcameras = j["cameras"]
    except KeyError:
        parseError("could not read cameras")
        return False
    for camera in streamcameras:
        if not streamreadCameraConfig(camera):
            return False

    # switched cameras
    if "switched cameras" in j:
        for camera in j["switched cameras"]:
            if not streamreadSwitchedCameraConfig(camera):
                return False

    return True

def streamstartCamera(config):
    """Start running the camera."""
    print("Starting camera '{}' on {}".format(config.name, config.path))
    camera = UsbCamera(config.name, config.path)
    streamserver = CameraServer.startAutomaticCapture(camera=camera)

    camera.setConfigJson(json.dumps(config.config))
    camera.setConnectionStrategy(VideoSource.ConnectionStrategy.kConnectionKeepOpen)

    if config.streamConfig is not None:
        streamserver.setConfigJson(json.dumps(config.streamConfig))

    return camera

def streamstartSwitchedCamera(config):
    """Start running the switched camera."""
    print("Starting switched camera '{}' on {}".format(config.name, config.key))
    server = CameraServer.addSwitchedCamera(config.name)

    def listener(event):
        data = event.data
        if data is not None:
            value = data.value.value()
            if isinstance(value, int):
                if value >= 0 and value < len(cameras):
                    server.setSource(cameras[value])
            elif isinstance(value, float):
                i = int(value)
                if i >= 0 and i < len(cameras):
                    server.setSource(cameras[i])
            elif isinstance(value, str):
                for i in range(len(cameraConfigs)):
                    if value == cameraConfigs[i].name:
                        server.setSource(cameras[i])
                        break

    NetworkTableInstance.getDefault().addListener(
        NetworkTableInstance.getDefault().getEntry(config.key),
        EventFlags.kImmediate | EventFlags.kValueAll,
        listener)

    return server

def main():

    print("Hello")

    vision_type = get_type()
    camera_orientation = camera_upside_down()

    # start NetworkTables
    ntconnect = NTConnectType(NTConnectType.SERVER)    #use CLIENT when running with rio
    ntinst = NetworkTableInstance.getDefault()
    if ntconnect == NTConnectType.SERVER:
        ntinst.startServer()
    else:
        print("connect as client")
        ntinst.startClient4("raspberrypi910")
        ntinst.setServerTeam(910)
 
    # Wait for NetworkTables to start
    time.sleep(1)
    
    rio_time_ntt = NTGetDouble(ntinst.getDoubleTopic(RIO_TIME_TOPIC_NAME), 0, 0, 0)
    
    if ntconnect == NTConnectType.CLIENT:
        while rio_time_ntt.get() == 0:
            time.sleep(1)
            print("Waiting to receive data from Rio...")
        print("Received data from Rio")

    # Table for vision output information
    uptime_ntt = NTGetDouble(ntinst.getDoubleTopic("/Vision/Uptime"), 0, 0, -1)

    debug_cage_ntt =  NTGetBoolean(ntinst.getBooleanTopic("/Vision/Cage Debug Mode"), DEBUG_MODE_DEFAULT, DEBUG_MODE_DEFAULT, DEBUG_MODE_DEFAULT)
    debug_coral_ntt = NTGetBoolean(ntinst.getBooleanTopic("/Vision/Coral Debug Mode"), DEBUG_MODE_DEFAULT, DEBUG_MODE_DEFAULT, DEBUG_MODE_DEFAULT)

    cageconfigfile_ntt = NTGetString(ntinst.getStringTopic(CAGE_CONFIG_FILE_TOPIC_NAME), CAGE_CONFIG_FILE_DEFAULT,CAGE_CONFIG_FILE_DEFAULT, CAGE_CONFIG_FILE_DEFAULT)
    coralconfigfile_ntt = NTGetString(ntinst.getStringTopic(CORAL_CONFIG_FILE_TOPIC_NAME), CORAL_CONFIG_FILE_DEFAULT,CORAL_CONFIG_FILE_DEFAULT, CORAL_CONFIG_FILE_DEFAULT)

    configfilefail_ntt = NTGetBoolean(ntinst.getBooleanTopic("/Vision/Config File Fail"), False, False, False)

    cage_active_ntt = NTGetBoolean(ntinst.getBooleanTopic(CORAL_ACTIVE_TOPIC_NAME), True, True, True)
    coral_active_ntt = NTGetBoolean(ntinst.getBooleanTopic(CORAL_ACTIVE_TOPIC_NAME), True, True, True)

    coral_pose_data_bytes_ntt = NTGetRaw(ntinst, CORAL_POSE_DATA_RAW_TOPIC_NAME, None, None, None)
    cage_pose_data_bytes_ntt = NTGetRaw(ntinst, CAGE_POSE_DATA_RAW_TOPIC_NAME, None, None, None)

    coral_pose_data_string_header_ntt = NTGetString(ntinst.getStringTopic(CORAL_POSE_DATA_STRING_TOPIC_NAME_HEADER),"", "", "") 
    cage_pose_data_string_header_ntt = NTGetString(ntinst.getStringTopic(CAGE_POSE_DATA_STRING_TOPIC_NAME_HEADER),"", "", "")

    temp_coral_ntt = NTGetDouble(ntinst.getDoubleTopic(CORAL_PI_TEMP_TOPIC_NAME), 0, 0, 0)
    temp_cage_ntt = NTGetDouble(ntinst.getDoubleTopic(CAGE_PI_TEMP_TOPIC_NAME), 0, 0, 0)

    cage_min_h_red_ntt = NTGetDouble(ntinst.getDoubleTopic(CAGE_MIN_HUE_RED_TOPIC_NAME), CAGE_MIN_HUE_RED, CAGE_MIN_HUE_RED, CAGE_MIN_HUE_RED)
    cage_min_s_red_ntt = NTGetDouble(ntinst.getDoubleTopic(CAGE_MIN_SAT_RED_TOPIC_NAME), CAGE_MIN_SAT_RED, CAGE_MIN_SAT_RED, CAGE_MIN_SAT_RED)
    cage_min_v_red_ntt = NTGetDouble(ntinst.getDoubleTopic(CAGE_MIN_VAL_RED_TOPIC_NAME), CAGE_MIN_VAL_RED, CAGE_MIN_VAL_RED, CAGE_MIN_VAL_RED)
    cage_max_h_red_ntt = NTGetDouble(ntinst.getDoubleTopic(CAGE_MAX_HUE_RED_TOPIC_NAME), CAGE_MAX_HUE_RED, CAGE_MAX_HUE_RED, CAGE_MAX_HUE_RED)
    cage_max_s_red_ntt = NTGetDouble(ntinst.getDoubleTopic(CAGE_MAX_SAT_RED_TOPIC_NAME), CAGE_MAX_SAT_RED, CAGE_MAX_SAT_RED, CAGE_MAX_SAT_RED)
    cage_max_v_red_ntt = NTGetDouble(ntinst.getDoubleTopic(CAGE_MAX_VAL_RED_TOPIC_NAME), CAGE_MAX_VAL_RED, CAGE_MAX_VAL_RED, CAGE_MAX_VAL_RED)

    cage_min_h_blue_ntt = NTGetDouble(ntinst.getDoubleTopic(CAGE_MIN_HUE_BLUE_TOPIC_NAME), CAGE_MIN_HUE_BLUE, CAGE_MIN_HUE_BLUE, CAGE_MIN_HUE_BLUE)
    cage_min_s_blue_ntt = NTGetDouble(ntinst.getDoubleTopic(CAGE_MIN_SAT_BLUE_TOPIC_NAME), CAGE_MIN_SAT_BLUE, CAGE_MIN_SAT_BLUE, CAGE_MIN_SAT_BLUE)
    cage_min_v_blue_ntt = NTGetDouble(ntinst.getDoubleTopic(CAGE_MIN_VAL_BLUE_TOPIC_NAME), CAGE_MIN_VAL_BLUE, CAGE_MIN_VAL_BLUE, CAGE_MIN_VAL_BLUE)
    cage_max_h_blue_ntt = NTGetDouble(ntinst.getDoubleTopic(CAGE_MAX_HUE_BLUE_TOPIC_NAME), CAGE_MAX_HUE_BLUE, CAGE_MAX_HUE_BLUE, CAGE_MAX_HUE_BLUE)
    cage_max_s_blue_ntt = NTGetDouble(ntinst.getDoubleTopic(CAGE_MAX_SAT_BLUE_TOPIC_NAME), CAGE_MAX_SAT_BLUE, CAGE_MAX_SAT_BLUE, CAGE_MAX_SAT_BLUE)
    cage_max_v_blue_ntt = NTGetDouble(ntinst.getDoubleTopic(CAGE_MAX_VAL_BLUE_TOPIC_NAME), CAGE_MAX_VAL_BLUE, CAGE_MAX_VAL_BLUE, CAGE_MAX_VAL_BLUE)

    coral_min_h_ntt = NTGetDouble(ntinst.getDoubleTopic(CORAL_MIN_HUE_TOPIC_NAME), CORAL_MIN_HUE, CORAL_MIN_HUE, CORAL_MIN_HUE)
    coral_min_s_ntt = NTGetDouble(ntinst.getDoubleTopic(CORAL_MIN_SAT_TOPIC_NAME), CORAL_MIN_SAT, CORAL_MIN_SAT, CORAL_MIN_SAT)
    coral_min_v_ntt = NTGetDouble(ntinst.getDoubleTopic(CORAL_MIN_VAL_TOPIC_NAME), CORAL_MIN_VAL, CORAL_MIN_VAL, CORAL_MIN_VAL)
    coral_max_h_ntt = NTGetDouble(ntinst.getDoubleTopic(CORAL_MAX_HUE_TOPIC_NAME), CORAL_MAX_HUE, CORAL_MAX_HUE, CORAL_MAX_HUE)
    coral_max_s_ntt = NTGetDouble(ntinst.getDoubleTopic(CORAL_MAX_SAT_TOPIC_NAME), CORAL_MAX_SAT, CORAL_MAX_SAT, CORAL_MAX_SAT)
    coral_max_v_ntt = NTGetDouble(ntinst.getDoubleTopic(CORAL_MAX_VAL_TOPIC_NAME), CORAL_MAX_VAL, CORAL_MAX_VAL, CORAL_MAX_VAL)

    coral_enable_ntt = NTGetBoolean(ntinst.getBooleanTopic(CORAL_ENABLE_TOPIC_NAME), False, False, False)
    cage_enable_ntt = NTGetBoolean(ntinst.getBooleanTopic(CAGE_ENABLE_TOPIC_NAME), False, False, False)

    coral_angle_ntt = NTGetDouble(ntinst.getDoubleTopic(CORAL_ANGLE_TOPIC_NAME), 0.0, 0.0, 0.0)
    coral_record_data_ntt = NTGetBoolean(ntinst.getBooleanTopic(CORAL_RECORD_DATA_TOPIC_NAME), False, False, False)
    coral_distance_ntt = NTGetDouble(ntinst.getDoubleTopic("/Vision/Coral Distance"), 0.0, 0.0, 0.0)

    cage_angle_ntt = NTGetDouble(ntinst.getDoubleTopic(CAGE_ANGLE_TOPIC_NAME), 0.0, 0.0, 0.0)
    cage_record_data_ntt = NTGetBoolean(ntinst.getBooleanTopic(CAGE_RECORD_DATA_TOPIC_NAME), False, False, False)
    cage_distance_ntt = NTGetDouble(ntinst.getDoubleTopic("/Vision/Cage Distance"), 0.0, 0.0, 0.0)

    coral_brightness_ntt = NTGetDouble(ntinst.getDoubleTopic(CORAL_BRIGHTNESS_TOPIC_NAME), BRIGHTNESS_DEFAULT, BRIGHTNESS_DEFAULT, BRIGHTNESS_DEFAULT)
    coral_contrast_ntt = NTGetDouble(ntinst.getDoubleTopic(CORAL_CONTRAST_TOPIC_NAME),  CONTRAST_DEFAULT, CONTRAST_DEFAULT, CONTRAST_DEFAULT)
    coral_ae_ntt = NTGetBoolean(ntinst.getBooleanTopic(CORAL_AE_TOPIC_NAME), AE_DEFAULT, AE_DEFAULT, AE_DEFAULT)
    coral_exposure_ntt = NTGetDouble(ntinst.getDoubleTopic(CORAL_EXPOSURE_TOPIC_NAME), EXPOSURE_DEFAULT, EXPOSURE_DEFAULT, EXPOSURE_DEFAULT)
    coral_crop_y_ntt = NTGetDouble(ntinst.getDoubleTopic(CORAL_Y_CROP_TOPIC_NAME), 0, 0, 0)

    cage_brightness_ntt = NTGetDouble(ntinst.getDoubleTopic(CAGE_BRIGHTNESS_TOPIC_NAME), BRIGHTNESS_DEFAULT, BRIGHTNESS_DEFAULT, BRIGHTNESS_DEFAULT)
    cage_contrast_ntt = NTGetDouble(ntinst.getDoubleTopic(CAGE_CONTRAST_TOPIC_NAME),  CONTRAST_DEFAULT, CONTRAST_DEFAULT, CONTRAST_DEFAULT)
    cage_ae_ntt = NTGetBoolean(ntinst.getBooleanTopic(CAGE_AE_TOPIC_NAME), AE_DEFAULT, AE_DEFAULT, AE_DEFAULT)
    cage_exposure_ntt = NTGetDouble(ntinst.getDoubleTopic(CAGE_EXPOSURE_TOPIC_NAME), EXPOSURE_DEFAULT, EXPOSURE_DEFAULT, EXPOSURE_DEFAULT)
    cage_crop_y_ntt = NTGetDouble(ntinst.getDoubleTopic(CAGE_Y_CROP_TOPIC_NAME), 0, 0, 0)


    coral_config_savefile_ntt = NTGetBoolean(ntinst.getBooleanTopic("/Vision/Coral Config Save"), False, False, False)
    coral_camera_savefile_ntt = NTGetBoolean(ntinst.getBooleanTopic("/Vision/Coral Camera Save"), False, False, False)
    coral_camera_refresh_nt_ntt = NTGetBoolean(ntinst.getBooleanTopic("/Vision/Coral Camera Refresh Nt"), False, False, False)
    coral_y_offset_ntt =  NTGetDouble(ntinst.getDoubleTopic(CORAL_Y_OFFSET_TOPIC_NAME), 0, 0, 0)
    cage_y_offset_ntt =  NTGetDouble(ntinst.getDoubleTopic(CAGE_Y_OFFSET_TOPIC_NAME), 0, 0, 0)
    cage_config_savefile_ntt = NTGetBoolean(ntinst.getBooleanTopic("/Vision/Cage Config Save"), False, False, False)
    cage_camera_savefile_ntt = NTGetBoolean(ntinst.getBooleanTopic("/Vision/Cage Camera Save"), False, False, False)
    cage_camera_refresh_nt_ntt = NTGetBoolean(ntinst.getBooleanTopic("/Vision/Cage Camera Refresh Nt"), False, False, False)

    is_red_alliance = NTGetBoolean(ntinst.getBooleanTopic(ALLIANCE_TYPE_TOPIC_NAME), True, True, True)
    perp_ntt = NTGetBoolean(ntinst.getBooleanTopic(PERP_TOPIC_NAME), False, False, False)

    # use for file
    config_coral = configparser.ConfigParser()
    config_gen = configparser.ConfigParser()
    config_cage = configparser.ConfigParser()

    '''
    print('*****')
    parser= configparser.ConfigParser()
    parser.read('gen_config.ini')
    print('read gen config file:')
    print({'GENERAL': dict(parser['GENERAL'])})
    return(parser.get('GENERAL', 'type'))
    '''

    file_read_gen(config_gen, configfilefail_ntt)

    file_read_coral(config_coral, configfilefail_ntt)
    nt_update_corals(config_coral, coralconfigfile_ntt, \
        coral_min_h_ntt, coral_min_s_ntt, coral_min_v_ntt, coral_max_h_ntt, coral_max_s_ntt, coral_max_v_ntt, \
            coral_brightness_ntt, coral_contrast_ntt, coral_ae_ntt, \
            coral_exposure_ntt, coral_y_offset_ntt, coral_crop_y_ntt)

    file_read_cage(config_cage, configfilefail_ntt)
    nt_update_cages(config_cage, cageconfigfile_ntt, \
        cage_min_h_red_ntt, cage_min_s_red_ntt, cage_min_v_red_ntt, \
            cage_max_h_red_ntt, cage_max_s_red_ntt, cage_max_v_red_ntt, \
            cage_min_h_blue_ntt, cage_min_s_blue_ntt, cage_min_v_blue_ntt, \
            cage_max_h_blue_ntt, cage_max_s_blue_ntt, cage_max_v_blue_ntt, \
            cage_brightness_ntt, cage_contrast_ntt, cage_ae_ntt, cage_exposure_ntt, cage_y_offset_ntt, cage_crop_y_ntt)

    
    file_read_gen(config_gen, configfilefail_ntt)
    
    coral_min_h = int(config_coral.get('VISION', CORAL_MIN_HUE_TOPIC_NAME))
    coral_min_s = int(config_coral.get('VISION', CORAL_MIN_SAT_TOPIC_NAME))
    coral_min_v = int(config_coral.get('VISION', CORAL_MIN_VAL_TOPIC_NAME))
    coral_max_h = int(config_coral.get('VISION', CORAL_MAX_HUE_TOPIC_NAME))
    coral_max_s = int(config_coral.get('VISION', CORAL_MAX_SAT_TOPIC_NAME))
    coral_max_v = int(config_coral.get('VISION', CORAL_MAX_VAL_TOPIC_NAME))

    cage_min_h_red = int(config_cage.get('VISION', CAGE_MIN_HUE_RED_TOPIC_NAME))
    cage_min_s_red = int(config_cage.get('VISION', CAGE_MIN_SAT_RED_TOPIC_NAME))
    cage_min_v_red = int(config_cage.get('VISION', CAGE_MIN_VAL_RED_TOPIC_NAME))
    cage_max_h_red = int(config_cage.get('VISION', CAGE_MAX_HUE_RED_TOPIC_NAME))
    cage_max_s_red = int(config_cage.get('VISION', CAGE_MAX_SAT_RED_TOPIC_NAME))
    cage_max_v_red = int(config_cage.get('VISION', CAGE_MAX_VAL_RED_TOPIC_NAME))

    cage_min_h_blue = int(config_cage.get('VISION', CAGE_MIN_HUE_BLUE_TOPIC_NAME))
    cage_min_s_blue = int(config_cage.get('VISION', CAGE_MIN_SAT_BLUE_TOPIC_NAME))
    cage_min_v_blue = int(config_cage.get('VISION', CAGE_MIN_VAL_BLUE_TOPIC_NAME))
    cage_max_h_blue = int(config_cage.get('VISION', CAGE_MAX_HUE_BLUE_TOPIC_NAME))
    cage_max_s_blue = int(config_cage.get('VISION', CAGE_MAX_SAT_BLUE_TOPIC_NAME))
    cage_max_v_blue = int(config_cage.get('VISION', CAGE_MAX_VAL_BLUE_TOPIC_NAME))

    #nice
    #load camera settings set from web console
    with open('/boot/frc.json') as f:
        web_settings = json.load(f)
    cam_config = web_settings['cameras'][0]

    w = cam_config['width']
    h = cam_config['height']
    fps = cam_config['fps']

    picam2 = Picamera2()
    server = FrameServer(picam2)

    if USE_CAGE_CAM_STREAM is True:
        # load stream camera settings and start steaming
        streamreadConfig()

        # start stream camera
        # work around wpilibsuite/allwpilib#5055
        CameraServer.setSize(CameraServer.kSize160x120)
        for config in streamcameraConfigs:
            streamcameras.append(streamstartCamera(config))

        # start switched camera (optional)
        #for config in streamswitchedCameraConfigs:
        #    streamstartSwitchedCamera(config)
            
    print(f'{len(picam2.sensor_modes)} camera image sensor modes')
    print(picam2.sensor_modes[0])
    #picam2_config = picam2.create_preview_configuration(main={"size" : (w,h)})
    #picam2_config = picam2.create_preview_configuration(main={"size" : (w,h)}, raw=sensor_modes[0])
    picam2_config = picam2.create_video_configuration( {'size': (w, h), 'format' : 'RGB888'})
    
    picam2.set_controls({"FrameRate": fps})
    game_piece = get_type()
    if game_piece == "coral":
        picam2.set_controls({'Brightness': float(config_coral.get('VISION', CORAL_BRIGHTNESS_TOPIC_NAME))})
        picam2.set_controls({'Contrast': float(config_coral.get('VISION', CORAL_CONTRAST_TOPIC_NAME))})

    elif game_piece == "cage":
        picam2.set_controls({'Brightness': float(config_cage.get('VISION', CAGE_BRIGHTNESS_TOPIC_NAME))})
        picam2.set_controls({'Contrast': float(config_cage.get('VISION', CAGE_CONTRAST_TOPIC_NAME))})

    # AeFlickerPeriod from https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf
    # The period of the lighting cycle in microseconds. For example, for 50Hz mains
    # lighting the flicker occurs at 100Hz, so the
    # period would be 10000 microseconds.
    # but this doesn't work: 'libcamera._libcamera.controls' has no attribute 'AeFlickerModeEnum
    # picam2.set_controls({"FrameRate": fps}, \
    #   {"AeFlickerMode": controls.AeFlickerModeEnum.Manual}, \
    #   {"AeFlickerPeriod": 12000})

    server.start()
    picam2.start()
    time.sleep(3)
    
    
    #picam2.set_controls({'AeEnable': False})
    #picam2.set_controls({'AwbMode': controls.AwbModeEnum.Indoor})
    
    #picam2.set_controls({'AwbMode': controls.AeExposureModeEnum.Short}) # didn't change tag fps
    #picam2.set_controls({"ExposureTime": 25000, "AnalogueGain": 2.0})

    # (optional) Setup a CvSource. This will send images back to the Dashboard
   
    outputStreamCoral = CameraServer.putVideo("coral image", cam_config['width'], cam_config['height'])
    outputStreamCage = CameraServer.putVideo("cage image", cam_config['width'], cam_config['height'])
    outputMask = CameraServer.putVideo("mask image", cam_config['width'], cam_config['height'])

    image_num = 0
    image_counter = 0
    image_time_av_total = 0
    fps_av = 0
    fps_av_min = 24601
    fps_av_max = -1
    seconds = 0
    current_seconds = 0
    prev_seconds = 0
    temp_sec = 30

    coral_brightness = float(config_coral.get('VISION', CORAL_BRIGHTNESS_TOPIC_NAME))
    coral_contrast = float(config_coral.get('VISION', CORAL_CONTRAST_TOPIC_NAME))
    coral_ae_mode = bool(config_coral.get('VISION', CORAL_AE_TOPIC_NAME))

    cage_brightness = float(config_cage.get('VISION', CAGE_BRIGHTNESS_TOPIC_NAME))
    cage_contrast = float(config_cage.get('VISION', CAGE_CONTRAST_TOPIC_NAME))
    cage_ae_mode = bool(config_cage.get('VISION', CAGE_AE_TOPIC_NAME))
    
    cam_settings_changed = False

    CORAL_Y_OFFSET = int(config_coral.get('VISION', CORAL_Y_OFFSET_TOPIC_NAME))
    CAGE_Y_OFFSET = int(config_cage.get('VISION', CAGE_Y_OFFSET_TOPIC_NAME))
    CORAL_Y_CROP = int(config_coral.get('VISION', CORAL_Y_CROP_TOPIC_NAME))
    CAGE_Y_CROP = int(config_cage.get('VISION', CAGE_Y_CROP_TOPIC_NAME))

    pose_data_bytes_ntt = None
    pose_data_string_header_ntt = None
    distance_ntt = None
    angle_ntt = None
    output_stream_image = None
    output_stream_mask = None
    record_data_ntt = None
    record_data_file_name = None
    lookup_distance = None
    lookup_angle = None
    
    while True:

        rio_time = rio_time_ntt.get()
        current_seconds = time.time()
        time_check = False
        
        db_coral = debug_coral_ntt.get()
        db_cage =  debug_cage_ntt.get()

        if current_seconds - prev_seconds >= UPTIME_UPDATE_INTERVAL:
            prev_seconds = current_seconds
            seconds = seconds + 1
            temp_sec = temp_sec + 1
        
            uptime_ntt.set(seconds)
            time_check = True

            if game_piece == 'coral':

                if db_coral == True:

                    print(f'sec={seconds} corals: ave fps={round(fps_av,0)} fps min={round(fps_av_min,0)} fps max={round(fps_av_max,0)}')
                        #print(f'CAGE_Y_OFFSET={CAGE_Y_OFFSET}')
                    
                    c_b = float(coral_brightness_ntt.get())
                    if c_b != coral_brightness:

                        picam2.set_controls({'Brightness': float(c_b)})
                        config_coral.set('VISION', CORAL_BRIGHTNESS_TOPIC_NAME , str(c_b))
                        coral_brightness = c_b

                    c_c = float(coral_contrast_ntt.get())
                    if c_c != coral_contrast:

                        picam2.set_controls({'Contrast': float(c_c)})
                        config_coral.set('VISION', CORAL_CONTRAST_TOPIC_NAME , str(c_c))
                        coral_contrast = c_c

                    c_ae = bool(coral_ae_ntt.get())
                    if c_ae != coral_ae_mode:
                    
                        picam2.set_controls({'AeEnable': bool(c_ae)})
                        config_coral.set('VISION', CORAL_AE_TOPIC_NAME , str(c_ae))
                        coral_ae_mode = c_ae

                        if coral_ae_mode == False:
                            exp_time = int(round(coral_exposure_ntt.get(),0))
                            picam2.set_controls({"ExposureTime": \
                                exp_time, "AnalogueGain": 1.0})
                            config_coral.set('VISION', CORAL_EXPOSURE_TOPIC_NAME, str(exp_time))

                else:
                    print(f'{seconds}')

            if game_piece == 'cage':

                if db_cage == True:
                    
                    print(f'sec={seconds} cages: ave fps={round(fps_av,0)} fps min={round(fps_av_min,0)} fps max={round(fps_av_max,0)}')
                        #print(f'CAGE_Y_OFFSET={CAGE_Y_OFFSET}')

                    c_b = float(cage_brightness_ntt.get())
                    if c_b != cage_brightness:

                        picam2.set_controls({'Brightness': float(c_b)})
                        config_cage.set('VISION', CAGE_BRIGHTNESS_TOPIC_NAME , str(c_b))
                        cage_brightness = c_b

                    c_c = float(cage_contrast_ntt.get())
                    if c_c != cage_contrast:

                        picam2.set_controls({'Contrast': float(c_c)})
                        config_cage.set('VISION', CORAL_CONTRAST_TOPIC_NAME , str(c_c))
                        cage_contrast = c_c

                    c_ae = bool(cage_ae_ntt.get())
                    if c_ae != cage_ae_mode:
                    
                        picam2.set_controls({'AeEnable': bool(c_ae)})
                        config_coral.set('VISION', CAGE_AE_TOPIC_NAME , str(c_ae))
                        cage_ae_mode = c_ae

                        if cage_ae_mode == False:
                            exp_time = int(round(cage_exposure_ntt.get(),0))
                            picam2.set_controls({"ExposureTime": \
                                exp_time, "AnalogueGain": 1.0})
                            config_cage.set('VISION', CAGE_EXPOSURE_TOPIC_NAME, str(exp_time))

                else:
                    print(f'{seconds}')                

        if temp_sec >= TEMP_UPDATE_INTERVAL:
            with open("/sys/class/thermal/thermal_zone0/temp", 'r') as f:
                current_temp = int(f.readline()) / 1000 #converting milidegrees C to degrees C
                temp_coral_ntt.set(current_temp)
                 
        t1_time = time.perf_counter()
        #img = picam2.capture_array()
        img = None
        img = server.wait_for_frame(img)
        #image_time = time.perf_counter() - t1_time
        # When the camera bolt hole is facing up, the camera is upside down
        # When the camera bolt hole is facing down, the camera is mounted right side up.
        # If flip needed, flip every image using cv2.flip(img,-1) 
        if camera_orientation == True:
            img = cv2.flip(img, -1)


        '''
        if frame_time == 0:
            # Send the output the error.
            outputStream.notifyError(cvSink.getError())
            # skip the rest of the current iteration
            continue
        '''
        
        #
        # Insert your image processing logic here!
          
        #CORAL!!!
        if game_piece == 'coral':
           
            if db_coral == True:
                coral_min_h = int(coral_min_h_ntt.get())
                coral_min_s = int(coral_min_s_ntt.get())
                coral_min_v = int(coral_min_v_ntt.get())
                coral_max_h = int(coral_max_h_ntt.get())
                coral_max_s = int(coral_max_s_ntt.get())
                coral_max_v = int(coral_max_v_ntt.get())
                CORAL_Y_OFFSET = int(coral_y_offset_ntt.get())
                CORAL_Y_CROP = int(coral_crop_y_ntt.get())

                if coral_config_savefile_ntt.get() == True:
                    file_write_corals(coralconfigfile_ntt.get(), \
                        coral_min_h_ntt.get(), \
                        coral_min_s_ntt.get(), \
                        coral_min_v_ntt.get(), \
                        coral_max_h_ntt.get(), \
                        coral_max_s_ntt.get(), \
                        coral_max_v_ntt.get(), \
                        coral_contrast_ntt.get(), \
                        coral_ae_ntt.get(), \
                        coral_exposure_ntt.get(), \
                        coral_y_offset_ntt.get(), \
                        coral_brightness_ntt.get(), \
                        coral_crop_y_ntt.get())
                    coral_config_savefile_ntt.set(False)

            min_h = coral_min_h 
            min_s = coral_min_s 
            min_v = coral_min_v 
            max_h = coral_max_h 
            max_s = coral_max_s 
            max_v = coral_max_v 
            Y_OFFSET = CORAL_Y_OFFSET
            Y_CROP = CORAL_Y_CROP
            X_OFFSET = CORAL_X_OFFSET
            min_area = CORAL_MIN_AREA
            min_center_y = CORAL_MIN_CENTER_Y
            max_center_y = CORAL_MAX_CENTER_Y
            min_aspect_ratio = CORAL_MIN_ASPECT_RATIO
            max_aspect_ratio = CORAL_MAX_ASPECT_RATIO
            use_extent = False
            use_extent_as_perp = False
            min_distance = CORAL_MIN_DISTANCE
            max_distance = CORAL_MAX_DISTANCE
            min_angle = CORAL_MIN_ANGLE
            max_angle = CORAL_MAX_ANGLE
            perp = False
            show_extreme_points = False
            adjust_distance_and_angle = False

            pose_data_bytes_ntt=coral_pose_data_bytes_ntt
            pose_data_string_header_ntt = coral_pose_data_string_header_ntt
            distance_ntt = coral_distance_ntt
            angle_ntt = coral_angle_ntt
            output_stream_image = outputStreamCoral
            output_stream_mask = outputMask
            record_data_ntt = coral_config_savefile_ntt
            record_data_file_name = 'coral_data.txt'
            find_distance = coral_regress_distance
            find_pixels_per_degree = coral_regress_px_per_deg
            piece_pose_data_bytes = coral_pose_data_bytes
            piece_pose_data_bytes_string = coral_pose_data_string

        elif game_piece == 'cage':

            if db_cage == True:
                cage_min_h_red = int(cage_min_h_red_ntt.get())
                cage_min_s_red = int(cage_min_s_red_ntt.get())
                cage_min_v_red = int(cage_min_v_red_ntt.get())
                cage_max_h_red = int(cage_max_h_red_ntt.get())
                cage_max_s_red = int(cage_max_s_red_ntt.get())
                cage_max_v_red = int(cage_max_v_red_ntt.get())
                cage_min_h_blue = int(cage_min_h_blue_ntt.get())
                cage_min_s_blue = int(cage_min_s_blue_ntt.get())
                cage_min_v_blue = int(cage_min_v_blue_ntt.get())
                cage_max_h_blue = int(cage_max_h_blue_ntt.get())
                cage_max_s_blue = int(cage_max_s_blue_ntt.get())
                cage_max_v_blue = int(cage_max_v_blue_ntt.get())
                CAGE_Y_OFFSET = int(cage_y_offset_ntt.get())
                CAGE_Y_CROP = int(cage_crop_y_ntt.get())

                if cage_config_savefile_ntt.get() == True:
                    print('check_button_press')
                    file_write_cages(cageconfigfile_ntt.get(), \
                        cage_min_h_red_ntt.get(), \
                        cage_min_s_red_ntt.get(), \
                        cage_min_v_red_ntt.get(), \
                        cage_max_h_red_ntt.get(), \
                        cage_max_s_red_ntt.get(), \
                        cage_max_v_red_ntt.get(), \
                        cage_min_h_blue_ntt.get(), \
                        cage_min_s_blue_ntt.get(), \
                        cage_min_v_blue_ntt.get(), \
                        cage_max_h_blue_ntt.get(), \
                        cage_max_s_blue_ntt.get(), \
                        cage_max_v_blue_ntt.get(), \
                        cage_contrast_ntt.get(), \
                        cage_ae_ntt.get(), \
                        cage_exposure_ntt.get(), \
                        cage_y_offset_ntt.get(), \
                        cage_brightness_ntt.get(), \
                        cage_crop_y_ntt.get())
                    cage_config_savefile_ntt.set(False)


            if is_red_alliance.get() == True:
                #print("loading red")
                min_h = cage_min_h_red
                min_s = cage_min_s_red 
                min_v = cage_min_v_red 
                max_h = cage_max_h_red 
                max_s = cage_max_s_red 
                max_v = cage_max_v_red
            else:
                #print("loading blue")
                min_h = cage_min_h_blue
                min_s = cage_min_s_blue 
                min_v = cage_min_v_blue 
                max_h = cage_max_h_blue 
                max_s = cage_max_s_blue 
                max_v = cage_max_v_blue

            Y_OFFSET = CAGE_Y_OFFSET
            Y_CROP = CAGE_Y_CROP
            X_OFFSET = CAGE_X_OFFSET
            min_area = CAGE_MIN_AREA
            min_center_y = CAGE_MIN_CENTER_Y
            max_center_y = CAGE_MAX_CENTER_Y
            min_aspect_ratio = CAGE_MIN_ASPECT_RATIO
            max_aspect_ratio = CAGE_MAX_ASPECT_RATIO
            use_extent = True
            min_extent = CAGE_MIN_EXTENT
            max_extent = CAGE_MAX_EXTENT
            min_distance = CAGE_MIN_DISTANCE
            max_distance = CAGE_MAX_DISTANCE
            min_angle = CAGE_MIN_ANGLE
            max_angle = CAGE_MAX_ANGLE
            use_extent_as_perp = False
            perp = False
            show_extreme_points = False
            adjust_distance_and_angle = True
            distance_adjustment = CAGE_DISTANCE_ADJUSTMENT
            angle_adjustment = CAGE_ANGLE_ADJUSTMENT

            pose_data_bytes_ntt=cage_pose_data_bytes_ntt
            pose_data_string_header_ntt = cage_pose_data_string_header_ntt
            distance_ntt = cage_distance_ntt
            angle_ntt = cage_angle_ntt
            output_stream_image = outputStreamCage
            output_stream_mask = outputMask
            record_data_ntt = cage_config_savefile_ntt
            record_data_file_name = 'cage_data.txt'
            find_distance = cage_regress_distance # change to the cage version once we have one
            find_pixels_per_degree = cage_regress_px_per_deg # change to the cage version once we have one
            piece_pose_data_bytes = cage_pose_data_bytes
            piece_pose_data_bytes_string = cage_pose_data_string

        # even though image capture format is RGB888, images are stored as BGR
        # for HSV filtering / masking / detecting, convert input image from BGR to HSV
        # but for displaying the image, convert input image from BGR to RGB
        original_image = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        img[0:Y_CROP, 0:w-1] = 0 # crop y axis image using slider user can change
        img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        color_low = np.array([min_h, min_s, min_v])
        color_high = np.array([max_h, max_s, max_v])
        img_mask = cv2.inRange(img_HSV, color_low, color_high)

        contours, useless = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #sorting the white pixels from largest to smallest
        contours_sorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        
        piece = []
        piece_contours = []
        
        max_contour = None
        center_y_max = -24
        area = 1
   
        if len(contours_sorted) >= 1:
            max_contour = contours_sorted[0]
        if max_contour is not None:

            area = cv2.contourArea(max_contour)
            if area > min_area:

                r_x,r_y,r_w,r_h = cv2.boundingRect(max_contour)
                center_x = r_x + int(round(r_w / 2))
                center_y = r_y + int(round(r_h / 2))
                # if adjust_center is True:
                #     if img_mask[center_y, center_x] > 0:
                #         while img_mask[center_y, center_x] > 0:
                #             center_x = center_x  + CAGE_CENTER_OFFSET
                #         center_x = center_x + CAGE_CENTER_OFFSET
                center_x += X_OFFSET
                center_y += Y_OFFSET
                extent = float(area) / (r_w * r_h)
                #print(f'ar={area:4.1f} ex={extent:1.2f} coral_x={center_x} coral_y={center_y}')
                              
        # at this point, max_contour points to closest shape by area or None if the area of all were too small
        # now need to determine if this shape is a coral
        if max_contour is not None:

            area = cv2.contourArea(max_contour)

            r_x,r_y,r_w,r_h = cv2.boundingRect(max_contour)
            center_x = r_x + int(round(r_w / 2)) + X_OFFSET
            center_y = r_y + int(round(r_h / 2)) + Y_OFFSET
            #print(f'center_x = {center_x} center_y = {center_y}')
            
            if (center_y > min_center_y  and center_y < 240*2):

                if (center_y > CAGE_CENTER_Y_CLOSE): # at really close, can't see the bottom, aspect ratio goes way up 
                    max_extent = CAGE_MAX_EXTENT_CLOSE
                    min_extent = CAGE_MIN_EXTENT_CLOSE
                    max_aspect_ratio = CAGE_MAX_AR_CLOSE
                # else:
                #     extent_min = 0.25

                #Extent is the ratio of contour area to bounding rectangle area.
                extent = float(area) / (r_w * r_h)


                aspect_ratio = (r_w/r_h)
                #print(f'aspect ratio = {aspect_ratio}')
                
                #print(f'AR={aspect_ratio:4.1f} area={area:4.1f} ex={extent:1.2f} x_pixel={center_x}, y_pixel={center_y}')

                #extent goes way down when we get real close
                if (aspect_ratio > min_aspect_ratio and aspect_ratio < max_aspect_ratio and use_extent is True and \
                    extent > min_extent and extent < max_extent):

                    print(f'AR={aspect_ratio:4.1f} area={area:4.1f} ex={extent:1.2f} x_pixel={center_x}, y_pixel={center_y}')

                # won't see full game piece this close, so y value for this distance is a bit off so force it to 0
                    if center_y >= max_center_y: 
                        distance = 0
                    else:
                        distance = find_distance(center_y) # get distance (inches) using y location subtracted by given value to account for bumpers
                    
                    rect = cv2.minAreaRect(max_contour)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    angle = 0
                    rot = 0

                    angle = ((center_x - w/2) * (1/find_pixels_per_degree(distance)))

                    if use_extent_as_perp is True:
                    
                        if extent > CAGE_MIN_EXTENT_PERP and extent < CAGE_MAX_EXTENT_PERP:
                            perp = True
                        else:
                            perp = False
                    else:
                        rot = rect[2]

                    # rect[2] is float angle in degrees
                    # print(f'distance={distance:4.1f}, y distance={center_y}, area={area:3.3f}, aspect ratio={aspect_ratio:3.3f}, angle = {(90-angle):3.3f}')
                    #distance=0
                    
                    if (distance >= min_distance and distance < max_distance) and (angle >= min_angle and angle < max_angle): # sanity check'''
                        
                        if adjust_distance_and_angle is True:

                            if img_mask[center_y,center_x] > 0:
                                
                                print("corrected")
                                distance += distance_adjustment
                                if angle > 0:
                                    angle += angle_adjustment
                                else:
                                    angle += (-1) * angle_adjustment

                        image_num += 1
                        image_counter += 1
                        image_time = time.perf_counter() - t1_time
                        image_time_av_total += image_time

                        if image_counter == FPS_NUM_SAMPLES:
                            fps_av = 1/(image_time_av_total/image_counter)
                            if fps_av < fps_av_min:
                                fps_av_min = fps_av
                            if fps_av > fps_av_max:
                                fps_av_max = fps_av
                            image_time_av_total = 0
                            image_counter = 0

                        if game_piece == 'cage':
                            pose_data = piece_pose_data_bytes(image_num, rio_time, image_time, 3, distance, angle, perp)
                        else:
                            pose_data = piece_pose_data_bytes(image_num, rio_time, image_time, 3, distance, angle, rot)

                        pose_data_bytes_ntt.set(pose_data)
                        NetworkTableInstance.getDefault().flush()

                        if db_coral or db_cage:
                           
                            if game_piece == 'cage':
                                txt = piece_pose_data_bytes_string(image_num, rio_time, image_time, distance, angle, perp)
                            else:
                                txt = piece_pose_data_bytes_string(image_num, rio_time, image_time, distance, angle, rot)

                            pose_data_string_header_ntt.set(txt)
                            distance_ntt.set(round(distance,2))
                            angle_ntt.set(round(angle,2))
                            perp_ntt.set(perp)
                            leftmost = tuple(max_contour[max_contour[:,:,0].argmin()][0])
                            rightmost = tuple(max_contour[max_contour[:,:,0].argmax()][0])
                            topmost = tuple(max_contour[max_contour[:,:,1].argmin()][0])
                            bottommost = tuple(max_contour[max_contour[:,:,1].argmax()][0])
                            if show_extreme_points is True:           
                                cv2.circle(original_image, (leftmost), 12, (200,0,0), -1)
                                cv2.circle(original_image, (rightmost), 12, (200,0,0), -1)
                                cv2.circle(original_image, (topmost), 12, (200,0,0), -1)
                                cv2.circle(original_image, (bottommost), 12, (200,0,0), -1)
                            cv2.circle(original_image, (center_x, center_y), 12, (200,200,0), -1)
                            cv2.drawContours(original_image, [box], 0, (0,200,0), 4)
                            output_stream_image.putFrame(original_image) # send to dashboard
                            output_stream_mask.putFrame(img_mask) # send to dashboard
                            if record_data_ntt.get() == True:
                                file_data = f'{area:4.1f},{extent:2.1f},{center_x},{center_y},{distance:3.1f},{angle:2.1f}'
                                with open(record_data_file_name, 'a') as f:
                                    f.write(file_data)
                                    f.write('\n')
                                record_data_ntt.set(False)
                            continue

        output_stream_image.putFrame(original_image) # send to dashboard
        output_stream_mask.putFrame(img_mask) # send to dashboard            

main()