import ffmpeg

(
    ffmpeg
    .input('Global_Changes/episode_5.MOV')
    .output('Global_Changes/episode_5_720p_10fps_1min.mp4', 
           an=None,  # remove audio
           r=10,     # set frame rate to 10
           t=60,     # limit duration to 60 seconds (1 minute)
           vf='scale=-1:720',  # scale to 720p
           vcodec='libx264',  # use H.264 codec
           crf=28,  # Constant Rate Factor (lower = better quality, higher = smaller size)
           preset='medium',  # encoding speed vs compression ratio
           movflags='+faststart'  # enable fast start for web playback
    )
    .run()
)