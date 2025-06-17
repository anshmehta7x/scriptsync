import cv2
from frames import FrameProcessor
from translator import TranslationCache
from ocr import OCRFrame

class Pipeline:
    def __init__(self, video_path, source_lang='en', dest_lang='fr', output_path='translated_video.mp4', 
                 difference_threshold=0.05, max_consecutive_skips=30):
        self.video_path = video_path
        self.source_lang = source_lang
        self.dest_lang = dest_lang
        self.output_path = output_path
        self.max_consecutive_skips = max_consecutive_skips  # Force processing after N skipped frames
        
        self.frame_processor = FrameProcessor(video_path, source_lang, difference_threshold)
        self.translator = TranslationCache()
        self.translator.set_source(source_lang)
        self.translator.set_dest(dest_lang)
        
        # Tracking variables
        self.last_processed_frame = None
        self.last_ocr_result = None
        self.last_translations = {}
        self.consecutive_skips = 0
        self.processing_stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'skipped_frames': 0,
            'forced_processing': 0
        }
        
    def process_video(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            self.output_path, 
            fourcc, 
            self.frame_processor.fps, 
            (self.frame_processor.width, self.frame_processor.height)
        )
        
        total_frames = self.frame_processor.frame_count
        
        for frame, should_process, diff_score in self.frame_processor.frame_generator():
            self.processing_stats['total_frames'] += 1
            frame_count = self.processing_stats['total_frames']
            
            # Force processing if we've skipped too many consecutive frames
            force_process = self.consecutive_skips >= self.max_consecutive_skips
            
            if should_process or force_process:
                if force_process:
                    self.processing_stats['forced_processing'] += 1
                    print(f"Frame {frame_count}: Forced processing after {self.consecutive_skips} skips")
                
                # Process current frame
                ocr_frame = OCRFrame(frame, self.source_lang)
                
                translations = {}
                for result in ocr_frame.get_results():
                    if result.text.strip():
                        try:
                            translated = self.translator.translate(result.text.strip())
                            translations[result.text] = translated
                        except Exception as e:
                            print(f"Translation error for '{result.text}': {e}")
                            translations[result.text] = result.text
                
                processed_frame = self.frame_processor.paste_frame(frame, ocr_frame, translations)
                
                # Update last processed data
                self.last_processed_frame = processed_frame.copy()
                self.last_ocr_result = ocr_frame
                self.last_translations = translations.copy()
                self.consecutive_skips = 0
                self.processing_stats['processed_frames'] += 1
                
                out.write(processed_frame)
                
                if frame_count % 30 == 0:
                    skip_percentage = (self.processing_stats['skipped_frames'] / frame_count) * 100
                    print(f"Processed {frame_count}/{total_frames} frames "
                          f"(Skipped: {self.processing_stats['skipped_frames']}, "
                          f"Skip%: {skip_percentage:.1f}%, "
                          f"Diff: {diff_score:.4f})")
                    
            else:
                # Skip processing - use last processed frame
                self.consecutive_skips += 1
                self.processing_stats['skipped_frames'] += 1
                
                if self.last_processed_frame is not None:
                    # Use the last processed frame with translations intact
                    out.write(self.last_processed_frame)
                else:
                    # Fallback: write original frame if no previous processed frame exists
                    out.write(frame)
                
                if frame_count % 100 == 0:  # Less frequent logging for skipped frames
                    skip_percentage = (self.processing_stats['skipped_frames'] / frame_count) * 100
                    print(f"Frame {frame_count}/{total_frames}: Skipped "
                          f"(Consecutive: {self.consecutive_skips}, "
                          f"Total Skip%: {skip_percentage:.1f}%, "
                          f"Diff: {diff_score:.4f})")
        
        out.release()
        self.print_final_statistics()
        print(f"Translation complete. Output saved to: {self.output_path}")
    
    def print_final_statistics(self):
        """Print comprehensive processing statistics"""
        stats = self.processing_stats
        total = stats['total_frames']
        processed = stats['processed_frames']
        skipped = stats['skipped_frames']
        forced = stats['forced_processing']
        
        if total > 0:
            processed_percentage = (processed / total) * 100
            skipped_percentage = (skipped / total) * 100
            forced_percentage = (forced / processed) * 100 if processed > 0 else 0
            
            print("\n" + "="*60)
            print("PROCESSING STATISTICS")
            print("="*60)
            print(f"Total frames:           {total:,}")
            print(f"Processed frames:       {processed:,} ({processed_percentage:.1f}%)")
            print(f"Skipped frames:         {skipped:,} ({skipped_percentage:.1f}%)")
            print(f"Forced processing:      {forced:,} ({forced_percentage:.1f}% of processed)")
            print(f"Difference threshold:   {self.frame_processor.difference_threshold}")
            print(f"Max consecutive skips:  {self.max_consecutive_skips}")
            
            # Calculate estimated time savings
            time_saved_percentage = skipped_percentage
            estimated_original_time = total  # Assume 1 time unit per frame originally
            estimated_new_time = processed
            time_savings = estimated_original_time - estimated_new_time
            
            print(f"\nEstimated processing time reduction: {time_saved_percentage:.1f}%")
            print(f"Performance improvement: {total/processed:.1f}x faster" if processed > 0 else "N/A")
            print("="*60)

    def get_statistics(self):
        """Return processing statistics dictionary"""
        return self.processing_stats.copy()

if __name__ == "__main__":
    
    # Conservative optimization (higher threshold = less skipping)
    conservative_pipeline = Pipeline(
        video_path="test_video.mp4",
        source_lang='fr',
        dest_lang='en',
        output_path='animals_english.mp4',
        difference_threshold=0.08,  # Higher threshold
        max_consecutive_skips=100000000    # Force processing sooner
    )
    
    # # Aggressive optimization (lower threshold = more skipping)
    # aggressive_pipeline = Pipeline(
    #     video_path="hack.mp4",
    #     source_lang='en',
    #     dest_lang='es',
    #     output_path='translated_aggressive.mp4',
    #     difference_threshold=0.05,  # Lower threshold
    #     max_consecutive_skips=45    # Allow more consecutive skips
    # )
    
    # Choose your optimization level
    pipeline = conservative_pipeline  # or aggressive_pipeline
    
    print(f"Starting video processing with difference threshold: {pipeline.frame_processor.difference_threshold}")
    print(f"Max consecutive skips: {pipeline.max_consecutive_skips}")
    
    pipeline.process_video()