import os
import cv2
import question_2


def read_video(path):
    return cv2.VideoCapture(path)


def video_to_frames(video: cv2.VideoCapture, out_path, start=0, end=-1, step=24):
    if not video.isOpened():
        print('No video')
        return

    video.set(cv2.CAP_PROP_POS_FRAMES, start)

    if end == -1:
        end = video.get(cv2.CAP_PROP_FRAME_COUNT)

    paths = []
    f_count = 0
    while video.isOpened() and video.get(cv2.CAP_PROP_POS_FRAMES) < end:
        _, img = video.read()
        path = os.path.join(out_path, f'video-{f_count}.png')
        question_2.write_image(img, path)
        paths.append(path)
        f_count += 1
        current_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
        video.set(cv2.CAP_PROP_POS_FRAMES, current_frame + step)

    return paths


def frames_to_video(paths, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=30, frame_sz=(600, 400),
                    out_path='../outputs/Assignment1/question_4/video/out.mp4'):
    out = cv2.VideoWriter(out_path, fourcc, fps, frame_sz)

    for path in paths:
        out.write(cv2.imread(path))

    out.release()


def main():
    vid = read_video('../res/Video1.avi')
    paths = video_to_frames(vid, '../outputs/Assignment1/question_4/frames', end=500, step=1)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 25
    frames_to_video(paths, fps=fps, frame_sz=(width, height))
    vid.release()


if __name__ == '__main__':
    main()
