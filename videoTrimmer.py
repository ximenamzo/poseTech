import ffmpeg

input_file = ffmpeg.input('danceVideos/mov/IMG_9174.mov')
output_file = ffmpeg.output(input_file.trim(start_frame=0, end_frame=200), 'output.mp4')
ffmpeg.run(output_file)