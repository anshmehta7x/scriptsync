import os
import cv2
import numpy as np
import easyocr
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import translation
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
cache = {}
fps = 0

class NoFrameError(Exception):
    pass

def get_frames(video_path):
    """
    Extract frames from a video file.

    Args:
        video_path (str): Path to the video file.

    Returns:
        np.array: Array of video frames.
    """
    global fps  # Declare fps as global
    capture = cv2.VideoCapture(video_path)
    frames = []

    # Get total frame count
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))  # Initialize fps here

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
    """
    Perform OCR on a single frame.

    Args:
        frame (np.array): Input frame in numpy array format.
        reader (easyocr.Reader): OCR reader instance.

    Returns:
        list: List of OCR results with text and bounding box.
    """
    frame_read = reader.readtext(frame)
    result = []
    for detection in frame_read:
        result.append({'text': detection[1], 'bounding_box': detection[0]})
    return result

def create_translation_cache(texts):
    """
    Create translation cache for a list of texts.

    Args:
        texts (list): List of texts to translate.
    """
    global cache  # Use global cache variable
    cache = translation.create_translate_cache(texts)

def check_frame_difference(frame1, frame2, threshold=1):
    """
    Check if there is a significant difference between two frames.

    Args:
        frame1 (np.array): First frame.
        frame2 (np.array): Second frame.
        threshold (float): Threshold for mean difference (default is 1).

    Returns:
        bool: True if frames are different, False otherwise.
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.resize(gray1, (160, 120))
    gray2 = cv2.resize(gray2, (160, 120))
    
    diff = cv2.absdiff(gray1, gray2)
    mean_diff = np.mean(diff)
    
    return mean_diff > threshold

def apply_translation(frame, ocr_result):
    """
    Apply text translation to a frame.

    Args:
        frame (np.array): Input frame in numpy array format.
        ocr_result (list): List of OCR results with text and bounding box.

    Returns:
        np.array: Translated frame with text overlays.
    """
    global cache  # Use global cache variable
    translated_frame = frame.copy()
    font_path = '../fonts/arial-unicode-ms.ttf'  # Ensure correct path to font file

    for detection in ocr_result:
        text = detection['text']
        bounding_box = detection['bounding_box']
        
        top_left = tuple(map(int, bounding_box[0]))
        bottom_right = tuple(map(int, bounding_box[2]))
        
        bg_color = np.mean(
            frame[top_left[1]-1:top_left[1]+1, bottom_right[0]:bottom_right[0]+1], axis=(0, 1))
        bg_color = tuple(int(value) if not np.isnan(value) else 0 for value in bg_color)
        
        cv2.rectangle(translated_frame, top_left, bottom_right, bg_color, -1)
        
        pil_image = Image.fromarray(translated_frame)
        draw = ImageDraw.Draw(pil_image)
        
        text_height = bottom_right[1] - top_left[1]
        font_size = max(int(text_height / 2), 1)
        
        if text in cache:
            logger.info("Cache hit for text: %s", text)
            translated_text = cache[text]
        else:
            logger.info("Cache miss for text: %s", text)
            translated_text = translation.translate_text(text)
            cache[text] = translated_text  # Update global cache
        
        font = ImageFont.truetype(font_path, font_size)
        draw.text((top_left[0], top_left[1]), translated_text, font=font, fill=(
            255-bg_color[0], 255-bg_color[1], 255-bg_color[2]))
        
        translated_frame = np.array(pil_image)

    return translated_frame

def ocr_video(frames, langs=['en'], GPU=True, cache_creation=False):
    """
    Perform OCR on a video and optionally apply text translation.

    Args:
        frames (np.array): Array of video frames.
        langs (list): List of languages for OCR (default is ['en'] for English).
        GPU (bool): Whether to use GPU for OCR (default is True).
        cache_creation (bool): Whether to create translation cache (default is False).

    Returns:
        list: List of translated frames if cache_creation is False.
    """
    global cache  # Use global cache variable
    reader = easyocr.Reader(langs, gpu=GPU)
    ocr_results = []
    translated_frames = []
    prev_frame = None
    prev_translated_frame = None

    with tqdm(total=len(frames), desc="Performing OCR") as pbar:
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
                if not cache_creation:
                    translated_frames.append(prev_translated_frame)
            pbar.update(1)

    if cache_creation:
        create_translation_cache(ocr_results)
    else:
        return translated_frames

def save_video(frames, output_path):
    """
    Save a sequence of frames as a video file.

    Args:
        frames (list): List of frames in numpy array format.
        output_path (str): Path to save the output video.
    """
    global fps  # Use global fps variable
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

if __name__ == "__main__":
    video_path = "heavy_eng.mp4"
    logger.info("Getting frames from video...")
    frames = get_frames(video_path)  # Retrieve frames only
    logger.info("Total frames extracted: %d", len(frames))
    
    logger.info("Performing OCR and caching...")
    ocr_video(frames, cache_creation=True)
    logger.info("OCR and caching completed")
    
    logger.info("Performing translation...")
    translated_frames = ocr_video(frames, cache_creation=False)
    logger.info("Translation done")
    
    logger.info("Saving video...")
    save_video(translated_frames, "output.mp4")
    logger.info("Video saved")
