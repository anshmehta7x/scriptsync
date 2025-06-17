import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ocr import OCRFrame

class FrameProcessor:
    def __init__(self, video_path: str, source_lang: str = 'en', difference_threshold: float = 0.05):
        self.video_path = os.path.abspath(video_path)
        self.capture = cv2.VideoCapture(self.video_path)
        if not self.capture.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.source_lang = source_lang
        self.font_path = 'fonts/arial-unicode-ms.ttf'
        
        # Frame difference detection parameters
        self.difference_threshold = difference_threshold
        self.previous_frame = None
        self.previous_processed_frame = None
        self.frame_skip_count = 0
    
    def calculate_frame_difference(self, current_frame, previous_frame):
        """
        Calculate the difference between two frames using multiple methods
        Returns a normalized difference score between 0 and 1
        """
        if previous_frame is None:
            return 1.0  # First frame, always process
        
        # Resize frames to smaller size for faster comparison
        small_size = (160, 90)  # Reduce resolution for comparison
        current_small = cv2.resize(current_frame, small_size)
        previous_small = cv2.resize(previous_frame, small_size)
        
        # Convert to grayscale for comparison
        current_gray = cv2.cvtColor(current_small, cv2.COLOR_BGR2GRAY)
        previous_gray = cv2.cvtColor(previous_small, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Mean Squared Error
        mse = np.mean((current_gray - previous_gray) ** 2)
        mse_normalized = mse / (255 ** 2)  # Normalize to 0-1
        
        # Method 2: Structural Similarity (simplified)
        # Calculate mean and variance
        mu1, mu2 = np.mean(current_gray), np.mean(previous_gray)
        var1, var2 = np.var(current_gray), np.var(previous_gray)
        covariance = np.mean((current_gray - mu1) * (previous_gray - mu2))
        
        # Simplified SSIM calculation
        c1, c2 = 0.01, 0.03
        ssim = ((2 * mu1 * mu2 + c1) * (2 * covariance + c2)) / \
               ((mu1**2 + mu2**2 + c1) * (var1 + var2 + c2))
        ssim_diff = 1 - ssim  # Convert similarity to difference
        
        # Method 3: Histogram comparison
        hist1 = cv2.calcHist([current_gray], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([previous_gray], [0], None, [256], [0, 256])
        hist_diff = 1 - cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # Combine methods (weighted average)
        combined_difference = (0.4 * mse_normalized + 0.4 * ssim_diff + 0.2 * hist_diff)
        
        return min(1.0, max(0.0, combined_difference))
    
    def should_process_frame(self, frame):
        """
        Determine if the current frame should be processed based on difference threshold
        """
        if self.previous_frame is None:
            return True
        
        difference = self.calculate_frame_difference(frame, self.previous_frame)
        return difference > self.difference_threshold
    
    def frame_generator(self):
        """
        Generator that yields frames with processing decision
        Returns tuple: (frame, should_process, difference_score)
        """
        while True:
            ret, frame = self.capture.read()
            if not ret:
                break
            
            should_process = self.should_process_frame(frame)
            difference_score = 0.0 if self.previous_frame is None else \
                             self.calculate_frame_difference(frame, self.previous_frame)
            
            yield frame, should_process, difference_score
            
            # Update previous frame for next comparison
            self.previous_frame = frame.copy()
        
        self.capture.release()
    
    def ocr_frame(self, frame):
        ocr_frame = OCRFrame(frame, self.source_lang)
        return ocr_frame.to_dict()
    
    def paste_frame(self, frame, ocr_frame, translations):
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        
        for result in ocr_frame.get_results():
            original_text = result.text
            if original_text in translations:
                translated_text = translations[original_text]
                bbox = result.bbox
                
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                min_x, max_x = int(min(x_coords)), int(max(x_coords))
                min_y, max_y = int(min(y_coords)), int(max(y_coords))
                
                # Ensure coordinates are within frame bounds
                min_x = max(0, min_x)
                min_y = max(0, min_y)
                max_x = min(frame.shape[1], max_x)
                max_y = min(frame.shape[0], max_y)
                
                # Calculate box dimensions
                box_width = max_x - min_x
                box_height = max_y - min_y
                
                # Skip if box is too small
                if box_width <= 0 or box_height <= 0:
                    continue
                
                # Calculate appropriate font size to fill the box
                font_size = self._calculate_font_size(translated_text, box_width, box_height)
                
                try:
                    font = ImageFont.truetype(self.font_path, font_size)
                except:
                    font = ImageFont.load_default()
                
                # Calculate average color only if we have a valid region
                if max_x > min_x and max_y > min_y:
                    avg_color = np.mean(frame[min_y:max_y, min_x:max_x], axis=(0, 1)).astype(int)
                else:
                    avg_color = np.array([128, 128, 128])  # Default gray color
                
                # Create semi-transparent background
                bg_color = tuple(avg_color.tolist()) + (200,)  # More opaque background
                
                # Choose text color based on background brightness
                brightness = np.mean(avg_color)
                text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
                
                # Add padding to background rectangle
                padding = max(2, int(font_size * 0.1))
                draw.rectangle([min_x-padding, min_y-padding, max_x+padding, max_y+padding], fill=bg_color)
                
                # Center text in the box
                text_x, text_y = self._center_text_in_box(
                    draw, translated_text, font, min_x, min_y, box_width, box_height
                )
                
                draw.text((text_x, text_y), translated_text, font=font, fill=text_color)
        
        return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
    
    def _calculate_font_size(self, text, box_width, box_height, min_size=8, max_size=72):
        """
        Calculate the optimal font size to fit text within the given box dimensions
        """
        # Start with a reasonable font size based on box height
        font_size = max(min_size, int(box_height * 0.7))
        font_size = min(font_size, max_size)
        
        # Try to load font and measure text
        try:
            font = ImageFont.truetype(self.font_path, font_size)
        except:
            font = ImageFont.load_default()
            return min_size
        
        # Create a temporary draw object to measure text
        temp_img = Image.new('RGB', (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)
        
        # Binary search for optimal font size
        min_font = min_size
        max_font = font_size
        optimal_size = min_size
        
        for _ in range(10):  # Limit iterations
            current_size = (min_font + max_font) // 2
            
            try:
                test_font = ImageFont.truetype(self.font_path, current_size)
            except:
                break
            
            # Get text dimensions
            bbox = temp_draw.textbbox((0, 0), text, font=test_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Check if text fits with some margin
            margin_factor = 0.9  # Use 90% of available space
            if (text_width <= box_width * margin_factor and 
                text_height <= box_height * margin_factor):
                optimal_size = current_size
                min_font = current_size + 1
            else:
                max_font = current_size - 1
            
            if min_font > max_font:
                break
        
        return max(min_size, optimal_size)
    
    def _center_text_in_box(self, draw, text, font, box_x, box_y, box_width, box_height):
        """
        Calculate the position to center text within a bounding box
        """
        # Get text dimensions
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Calculate centered position
        text_x = box_x + (box_width - text_width) // 2
        text_y = box_y + (box_height - text_height) // 2
        
        # Ensure text doesn't go outside the box
        text_x = max(box_x, text_x)
        text_y = max(box_y, text_y)
        
        return text_x, text_y
    
    def get_statistics(self):
        """
        Return processing statistics
        """
        total_frames = self.frame_count
        processed_frames = total_frames - self.frame_skip_count
        skip_percentage = (self.frame_skip_count / total_frames * 100) if total_frames > 0 else 0
        
        return {
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'skipped_frames': self.frame_skip_count,
            'skip_percentage': skip_percentage,
            'difference_threshold': self.difference_threshold
        }

if __name__ == "__main__":
    video_path = "test_video.mp4"
    frame_processor = FrameProcessor(video_path, 'fr', difference_threshold=0.03)
    
    frame_count = 0
    for frame, should_process, diff_score in frame_processor.frame_generator():
        frame_count += 1
        print(f"Frame {frame_count}: Process={should_process}, Difference={diff_score:.4f}")
        
        if should_process:
            ocr_result = frame_processor.ocr_frame(frame)
            print(f"OCR Result: {ocr_result}")
        else:
            print("Frame skipped - using previous result")
            
        if frame_count >= 10:  # Test first 10 frames
            break
    
    stats = frame_processor.get_statistics()
    print(f"\nProcessing Statistics: {stats}")