import os
import cv2
import numpy as np
import easyocr
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

import translation

class NoFrameError(Exception):
    pass

def get_frames(video_path):
    capture = cv2.VideoCapture(video_path)
    frames = []

    # Get total frame count
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    global fps
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    

    with tqdm(total=total_frames, desc="Reading Frames") as pbar:
        while True:
            ret, frame = capture.read()
            if ret:
                frames.append(frame)
                pbar.update(1)
            else:
                break

    if not frames:
        raise NoFrameError('No frame found in video')

    capture.release()
    cv2.destroyAllWindows()

    frames_array = np.array(frames)
    return frames_array



def ocr_single_frame(frame, reader):
    frame_read = reader.readtext(frame)
    result = []
    for detection in frame_read:
        result.append({'text': detection[1], 'bounding_box': detection[0]})
    return result

def create_translation_cache(texts):
    global cache
    cache = translation.create_translate_cache(texts)

def check_frame_difference(frame1, frame2, threshold=1):
    # Convert to grayscale and downscale for faster processing
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.resize(gray1, (160, 120))
    gray2 = cv2.resize(gray2, (160, 120))
    
    # Compute absolute difference and mean
    diff = cv2.absdiff(gray1, gray2)
    mean_diff = np.mean(diff)
    
    return mean_diff > threshold

def apply_translation(frame, ocr_result):
    translated_frame = frame.copy()
    font_path = '../fonts/arial-unicode-ms.ttf'  # Make sure this path is correct

    for detection in ocr_result:
        text = detection['text']
        bounding_box = detection['bounding_box']
        
        # Extract coordinates
        top_left = tuple(map(int, bounding_box[0]))
        bottom_right = tuple(map(int, bounding_box[2]))
        
        # Calculate background color
        bg_color = np.mean(
            frame[top_left[1]-1:top_left[1]+1, bottom_right[0]:bottom_right[0]+1], axis=(0, 1))
        bg_color = tuple(int(value) if not np.isnan(value) else 0 for value in bg_color)
        
        # Draw rectangle to cover original text
        cv2.rectangle(translated_frame, top_left, bottom_right, bg_color, -1)
        
        # Prepare for text drawing
        pil_image = Image.fromarray(translated_frame)
        draw = ImageDraw.Draw(pil_image)
        
        # Calculate font size
        text_height = bottom_right[1] - top_left[1]
        font_size = max(int(text_height / 2), 1)
        
        # Translate text
        if 'cache' in globals() and text in cache:
            print("cache hit")
            translated_text = cache[text]
        else:
            print("cache miss")
            translated_text = translation.translate_text(text)

        
        # Draw translated text
        font = ImageFont.truetype(font_path, font_size)
        draw.text((top_left[0], top_left[1]), translated_text, font=font, fill=(
            255-bg_color[0], 255-bg_color[1], 255-bg_color[2]))
        
        # Convert back to numpy array
        translated_frame = np.array(pil_image)

    return translated_frame


def ocr_video(frames, langs=['en'], GPU=True, cache_creation=False):
    reader = easyocr.Reader(langs, gpu=GPU)
    ocr_results = []
    translated_frames = []
    prev_frame = None
    prev_translated_frame = None
    with tqdm(total=len(frames), desc="OCRing frames") as pbar:
        for frame in frames:
            if prev_frame is None or check_frame_difference(frame, prev_frame):
                ocr_result = ocr_single_frame(frame, reader)
                ocr_results.append(ocr_result)
                if not cache_creation:
                    translated_frame = apply_translation(frame, ocr_result)
                    translated_frames.append(translated_frame)
                    prev_translated_frame = translated_frame
                prev_frame = frame
            else:
                # Copy the previous translated frame if the current frame isn't different enough
                if not cache_creation:
                    translated_frames.append(prev_translated_frame)
            pbar.update(1)
    if cache_creation:
        create_translation_cache(ocr_results)
    else:
        return translated_frames

def save_video(frames, output_path, fps=30):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()




if __name__ == "__main__":
    video_path = "heavy_eng.mp4"
    print("Getting frames...")
    f = get_frames(video_path)
    print(f"Total frames: {len(f)}")
    
    print("Performing OCR and caching...")
    ocr_video(f, cache_creation=True)
    print("Cache created")
    print("Performing translation...")
    output = ocr_video(f, cache_creation=False)
    print("Translation done")
    print("Saving video...")
    save_video(output,"output.mp4", fps)
