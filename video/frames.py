import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ocr import OCRFrame

class FrameProcessor:
    def __init__(self, video_path: str, source_lang: str = 'en'):
        self.video_path = os.path.abspath(video_path)
        self.capture = cv2.VideoCapture(self.video_path)
        if not self.capture.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.source_lang = source_lang # Default language for OCR, can be changed later
        self.font_path = 'fonts/arial-unicode-ms.ttf'
    
    def frame_generator(self):
        while True:
            ret, frame = self.capture.read()
            if not ret:
                break
            yield frame
        self.capture.release()
    
    def ocr_frame(self, frame):
        ocr_frame = OCRFrame(frame, self.source_lang)
        return ocr_frame.to_dict()
    
    def paste_frame(self, frame, ocr_frame, translations):
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)

            
            try:
                font = ImageFont.truetype(self.font_path, 24)
            except:
                font = ImageFont.load_default()
            
            for result in ocr_frame.get_results():
                original_text = result.text
                if original_text in translations:
                    translated_text = translations[original_text]
                    bbox = result.bbox
                    
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    min_x, max_x = min(x_coords), max(x_coords)
                    min_y, max_y = min(y_coords), max(y_coords)
                    avg_color = np.mean(frame[min_y:max_y, min_x:max_x], axis=(0, 1)).astype(int)
                    bg_color = tuple(avg_color.tolist()) + (180,)
                    text_color = (255, 255, 255)
                    
                    draw.rectangle([min_x-5, min_y-5, max_x+5, max_y+5], fill=bg_color)
                    draw.text((min_x, min_y), translated_text, font=font, fill=text_color)
            
            return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    video_path = "test_video.mp4"
    frame_processor = FrameProcessor(video_path, 'fr')
    
    for frame in frame_processor.frame_generator():
        ocr_result = frame_processor.ocr_frame(frame)
        print(ocr_result)
        break