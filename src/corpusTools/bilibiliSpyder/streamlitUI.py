import os
import sys
import time
import threading
import streamlit as st
from videoCommentsManager import VideoCommentsManager

if __name__ == '__main__':

    commentsManager = VideoCommentsManager()
    commentsManager.displayScarpingPage()



