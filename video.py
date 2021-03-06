from utils import *
from find_lanes import *
from moviepy.editor import VideoFileClip
from IPython.display import HTML

test_output = 'test.mp4'
# clip = VideoFileClip('test_video.mp4')
clip = VideoFileClip('project_video.mp4')
lane_finder = LaneFinder()
test_clip=clip.fl_image(lane_finder.process_image)
test_clip.write_videofile(test_output, audio=False)
# print("all centers = {}".format(lane_finder.tracker.get_all_statistics()))