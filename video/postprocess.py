import subprocess

# requires ffmpeg to be installed and available in the system PATH

def compress_video_h265(input_path, output_path):
    command = [
        "ffmpeg",
        "-i", input_path,
        "-an",
        "-vcodec", "libx265",
        "-preset", "ultrafast",
        "-crf", "28",
        output_path
    ]
    subprocess.run(command, check=True)
