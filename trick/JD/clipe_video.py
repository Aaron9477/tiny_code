from moviepy.editor import *
import os


def clip(video_path, times, save_dir):
    v = VideoFileClip(video_path)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for i in range(0,len(times)):
        print(i,' begins...')
        if i == 0:
            CompositeVideoClip([v.subclip(0,times[i])]). \
                to_videofile(save_dir + "/" + save_dir + "_" + str(i) + ".mp4")
        else:
            CompositeVideoClip([v.subclip(times[i-1], times[i])]). \
                to_videofile(save_dir + "/" + save_dir + "_" + str(i) + ".mp4")
        print(i, ' done...')
    if len(times) > 1 and times[len(times)-1] < 57:
        CompositeVideoClip([v.subclip(times[len(times)-1], 58)]). \
            to_videofile(save_dir + "/" + save_dir + "_" + str(len(times)) + ".mp4")
        print(len(times), 'done...')


if __name__ == '__main__':
    # 只要中间几个点(秒为单位)的时间，下面的会切成0-3.5，3-15，...28-29，29-58
    # clip("E:/jd/raw_video/1.mp4",[3.5,15,20,25,28,29],"1")
    f = open("cut.txt")
    while 1:
        line = f.readline()
        if not line:
            break
        if line.startswith("#"):
            continue
        clip("E:/jd/raw_video/"+line.split(',')[0]+".mp4",
             [float(x) for x in line.split(',')[1:]],
             line.split(',')[0] )