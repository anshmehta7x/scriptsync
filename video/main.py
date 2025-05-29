import cv2
from frames import FrameProcessor
from translator import TranslationCache
from ocr import OCRFrame

class Pipeline:
    def __init__(self, video_path, source_lang='en', dest_lang='fr', output_path='translated_video.mp4'):
        self.video_path = video_path
        self.source_lang = source_lang
        self.dest_lang = dest_lang
        self.output_path = output_path
        
        self.frame_processor = FrameProcessor(video_path, source_lang)
        self.translator = TranslationCache()
        self.translator.set_source(source_lang)
        self.translator.set_dest(dest_lang)
        
    def process_video(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            self.output_path, 
            fourcc, 
            self.frame_processor.fps, 
            (self.frame_processor.width, self.frame_processor.height)
        )
        
        frame_count = 0
        total_frames = self.frame_processor.frame_count
        
        for frame in self.frame_processor.frame_generator():
            ocr_frame = OCRFrame(frame, self.source_lang)
            
            translations = {}
            for result in ocr_frame.get_results():
                if result.text.strip():
                    try:
                        translated = self.translator.translate(result.text.strip())
                        translations[result.text] = translated
                    except:
                        translations[result.text] = result.text
            
            processed_frame = self.frame_processor.paste_frame(frame, ocr_frame, translations)
            out.write(processed_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                
                print(f"Processed {frame_count}/{total_frames} frames")
                break
        
        out.release()
        print(f"Translation complete. Output saved to: {self.output_path}")

if __name__ == "__main__":
    pipeline = Pipeline(
        video_path="hack.mp4",
        source_lang='en',
        dest_lang='es',
        output_path='translated_output.mp4'
    )
    pipeline.process_video()