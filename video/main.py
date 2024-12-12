import frame_processor
import cv2

def batch_process_video(file, output_file="output.mp4", threshold=3, batch_size=100):
    cap = cv2.VideoCapture(file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    print("Total frames in video:", total_frames)
    current_frame = 0
    all_frames = []
    temp_dir = "temp_frames"
    unique_count = 0

    while current_frame < total_frames:
        batch = min(batch_size, total_frames - current_frame)

        frames = frame_processor.get_frames(file, batch, current_frame)
        if not frames:
            break

        frame_diffs = frame_processor.get_frame_diffs(frames)
        for i in frame_diffs:
            if i > threshold:
                unique_count += 1
        processed_frames = frame_processor.selective_batch_process(frames, frame_diffs, threshold)

        all_frames.extend(processed_frames)
        current_frame += batch

    frame_processor.stitch_frames_batch(all_frames, output_file, fps, temp_dir)
    print(f"Batch processing completed. Output saved to {output_file}")
    print("Unique frames:", unique_count)


if __name__ == "__main__":
    input_file = "french2_fr.mp4"
    #threshold value can be used to adjust skipped frame amount
    # Smaller batch size = less memory usage, but slower processing, higher quality though

    batch_process_video(input_file, output_file="output_video.mp4", threshold=0.5, batch_size=50)
