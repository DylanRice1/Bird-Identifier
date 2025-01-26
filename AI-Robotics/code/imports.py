import tensorflow as tf
import numpy as np
import pandas as pd
import os
import ultralytics
import roboflow
import tensorflow as tf
import keras_cv
import keras_core as keras
import tensorflow as tf
import splitfolders
import imghdr
import time
import matplotlib.pyplot as plt 
import itertools
import yolov8
import cv2
import re
import tkinter as tk

from ultralytics import YOLO
from IPython import display
from inference import get_model
from roboflow import Roboflow
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from keras.models import load_model
from tkinter import filedialog