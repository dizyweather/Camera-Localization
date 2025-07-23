# Import everything needed to edit video clips
from moviepy.editor import *


VIDEO_PATH = "Y:\\Swarm Assembly 2025\\S01\\0711\\camcorder\\C0009.MP4"
# loading video dsa gfg intro video
clip = VideoFileClip(VIDEO_PATH)

# clipping of the video 
# getting video for only starting 10 seconds
clip = clip.subclip(12, 12 + 20)

clip_without_audio = clip.without_audio()

# final = clip_without_audio.fx( vfx.speedx, 100)
final = clip_without_audio

#2min ~ 40min

final.write_videofile("swarm_knock_recover_no_audio.mp4")
