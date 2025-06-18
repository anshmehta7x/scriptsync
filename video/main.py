import cv2
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from typing import List, Tuple, Dict
import gc
import psutil
import os
from contextlib import contextmanager

# --- Project-Specific Imports ---
# These classes are expected to be in their respective files (e.g., ocr.py)
# Ensure these files are in your Python path.
from ocr import OCRFrame
from translator import TranslationCache
from frames import FrameProcessor


class FrameBatch:
    """Memory-efficient container for a batch of frames."""
    def __init__(self, batch_id: int, frame_indices: List[int],
                 start_frame_idx: int, end_frame_idx: int):
        self.batch_id = batch_id
        self.frame_indices = frame_indices
        self.start_frame_idx = start_frame_idx
        self.end_frame_idx = end_frame_idx

class MemoryOptimizedPipeline:
    def __init__(self, video_path, source_lang='en', dest_lang='fr', output_path='translated_video.mp4',
                 difference_threshold=0.05, max_consecutive_skips=30,
                 batch_size=100, max_workers=None, memory_limit_gb=4, gpu_concurrency_limit=2):
        """
        High-performance, memory-optimized pipeline for video translation.

        Args:
            batch_size: Number of frames to process in a single batch.
            max_workers: Max number of CPU threads. Defaults to a high number for performance.
            memory_limit_gb: RAM limit in GB for monitoring purposes.
            gpu_concurrency_limit: Max number of threads allowed to perform GPU operations concurrently.
        """
        self.video_path = video_path
        self.source_lang = source_lang
        self.dest_lang = dest_lang
        self.output_path = output_path
        self.difference_threshold = difference_threshold
        self.max_consecutive_skips = max_consecutive_skips
        self.batch_size = batch_size
        self.memory_limit_gb = memory_limit_gb
        self.gpu_concurrency_limit = gpu_concurrency_limit

        # Unlocks CPU performance, good for I/O-bound tasks like frame reading
        if max_workers is None:
            self.max_workers = min(32, (os.cpu_count() or 1) + 4)
        else:
            self.max_workers = max_workers

        # Semaphore to control concurrent access to the GPU
        self.gpu_semaphore = threading.Semaphore(self.gpu_concurrency_limit)

        print(f"🚀 Initializing pipeline: {self.max_workers} CPU workers, "
              f"{self.gpu_concurrency_limit} GPU workers, "
              f"{self.memory_limit_gb}GB limit, batch size {self.batch_size}")

        self._get_video_info()

        self.global_translation_cache = TranslationCache()
        self.global_translation_cache.set_source(source_lang)
        self.global_translation_cache.set_dest(dest_lang)
        self.cache_lock = threading.Lock()
        self.max_cache_size = 2000

        self.progress_lock = threading.Lock()
        self.frames_processed_count = 0
        self.last_progress_time = time.time()

        self.processing_stats = {
            'total_frames_written': 0,
            'batches_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_warnings': 0,
            'peak_memory_gb': 0.0
        }

    def _get_video_info(self):
        """Get video information and immediately close capture."""
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found at the specified path: {self.video_path}")

        temp_cap = cv2.VideoCapture(self.video_path)
        if not temp_cap.isOpened():
            raise IOError(f"Could not open video file: {self.video_path}")

        self.frame_count = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = temp_cap.get(cv2.CAP_PROP_FPS)
        self.width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        temp_cap.release()

        print(f"📹 Video info: {self.frame_count} frames, {self.fps:.1f} FPS, {self.width}x{self.height}")

    @contextmanager
    def memory_monitor(self, operation_name: str):
        """Context manager for monitoring memory usage."""
        start_memory = psutil.Process().memory_info().rss / (1024**3)
        try:
            yield
        finally:
            end_memory = psutil.Process().memory_info().rss / (1024**3)
            peak_memory = max(start_memory, end_memory)

            if peak_memory > self.processing_stats['peak_memory_gb']:
                self.processing_stats['peak_memory_gb'] = peak_memory

            if peak_memory > self.memory_limit_gb * 0.9:
                self.processing_stats['memory_warnings'] += 1
                print(f"⚠️ Memory warning in {operation_name}: {peak_memory:.2f}GB")
                self._emergency_cleanup()

    def _clear_cuda_cache(self):
        """Safely clear CUDA cache if PyTorch is available."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, Exception):
            pass # Ignore if torch not installed or fails

    def _emergency_cleanup(self):
        """Force garbage collection and cleanup."""
        print("🧹 Emergency memory cleanup...")
        self._clear_cuda_cache()
        gc.collect()

    def create_frame_batches(self) -> List[FrameBatch]:
        """Create lightweight batches with frame indices only."""
        batches = []
        for i in range(0, self.frame_count, self.batch_size):
            start_idx = i
            end_idx = min(start_idx + self.batch_size - 1, self.frame_count - 1)
            frame_indices = list(range(start_idx, end_idx + 1))
            batches.append(FrameBatch(len(batches), frame_indices, start_idx, end_idx))
        print(f"✅ Created {len(batches)} batches total.")
        return batches

    def update_progress(self, frames_in_batch: int):
        """Thread-safe progress updates."""
        with self.progress_lock:
            self.frames_processed_count += frames_in_batch
            current_time = time.time()
            if current_time - self.last_progress_time < 2.0:
                return

            elapsed_time = current_time - self.start_time
            if self.frames_processed_count > 0:
                fps = self.frames_processed_count / elapsed_time
                eta_seconds = (self.frame_count - self.frames_processed_count) / fps if fps > 0 else 0
                progress_percent = (self.frames_processed_count / self.frame_count) * 100
                current_memory = psutil.Process().memory_info().rss / (1024**3)

                print(f"⏳ Progress: {progress_percent:.1f}% ({self.frames_processed_count:,}/{self.frame_count:,}) "
                      f"| {fps:.1f} FPS | ETA: {eta_seconds/60:.1f}min | RAM: {current_memory:.2f}GB")
            self.last_progress_time = current_time

    def process_batch(self, batch: FrameBatch) -> Tuple[int, List[Tuple[int, np.ndarray]]]:
        """
        Process a single batch of frames. Reads frames from disk, performs OCR/translation,
        and returns the processed frames. GPU-intensive work is throttled by a semaphore.
        """
        processed_frames = []
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"❌ Thread {threading.get_ident()}: Failed to open video for batch {batch.batch_id}")
            return batch.batch_id, []

        try:
            last_processed_frame = None
            consecutive_skips = 0
            temp_processor = FrameProcessor(self.video_path, self.source_lang, self.difference_threshold)

            for frame_idx in batch.frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret: continue

                should_process = (consecutive_skips >= self.max_consecutive_skips)

                if not should_process:
                    if last_processed_frame is not None:
                        diff = cv2.absdiff(frame, last_processed_frame)
                        if np.mean(diff) / 255 > self.difference_threshold:
                            should_process = True
                    else:
                        should_process = True

                if should_process:
                    with self.gpu_semaphore:
                        try:
                            ocr_frame = OCRFrame(frame, self.source_lang)
                            translations = self._get_translations_cached(ocr_frame)
                            processed_frame = temp_processor.paste_frame(frame, ocr_frame, translations)
                            last_processed_frame = processed_frame.copy()
                            consecutive_skips = 0
                        except Exception as e:
                            print(f"❌ Error processing frame {frame_idx}: {e}")
                            processed_frame = frame
                    processed_frames.append((frame_idx, processed_frame))
                else:
                    consecutive_skips += 1
                    processed_frames.append((frame_idx, last_processed_frame.copy()))

            self.update_progress(len(batch.frame_indices))
            return batch.batch_id, processed_frames

        finally:
            cap.release()
            del temp_processor
            gc.collect()

    def _get_translations_cached(self, ocr_frame: OCRFrame) -> Dict[str, str]:
        """Get translations for all text in an OCRFrame, using a shared cache."""
        translations = {}
        hits, misses = 0, 0
        texts_to_translate = []
        
        with self.cache_lock:
            for result in ocr_frame.get_results():
                text = result.text.strip()
                if not text: continue
                cached = self.global_translation_cache.fuzzy_match_transaltions(text)
                if cached:
                    translations[text] = cached
                    hits += 1
                elif len(self.global_translation_cache.cache) < self.max_cache_size:
                    texts_to_translate.append(text)

        if texts_to_translate:
            for text in texts_to_translate:
                try:
                    translated = self.global_translation_cache.translate(text)
                    translations[text] = translated
                    misses += 1
                except Exception as e:
                    print(f"❌ Translation error for '{text}': {e}")
                    translations[text] = text

        with self.progress_lock:
            self.processing_stats['cache_hits'] += hits
            self.processing_stats['cache_misses'] += misses

        return translations

    def process_video(self):
        """Main processing function with concurrent processing and sequential writing."""
        print("🎬 Starting high-performance video processing...")
        self.start_time = time.time()
        batches = self.create_frame_batches()
        if not batches:
            print("❌ No frames to process!")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
        if not out.isOpened():
            print("❌ Failed to open video writer!")
            return

        completed_results = {}
        next_batch_to_write = 0

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_batch = {executor.submit(self.process_batch, batch): batch for batch in batches}

                for future in as_completed(future_to_batch):
                    try:
                        batch_id, processed_frames = future.result()
                        completed_results[batch_id] = processed_frames
                        self.processing_stats['batches_processed'] += 1

                        while next_batch_to_write in completed_results:
                            self._write_batch_frames(out, completed_results[next_batch_to_write])
                            del completed_results[next_batch_to_write]
                            next_batch_to_write += 1
                            gc.collect()

                    except Exception as e:
                        batch = future_to_batch[future]
                        print(f"❌ Batch {batch.batch_id} generated an exception: {e}")

            print("✅ All batches processed, writing any remaining frames...")
            while next_batch_to_write < len(batches):
                 if next_batch_to_write in completed_results:
                    self._write_batch_frames(out, completed_results[next_batch_to_write])
                    del completed_results[next_batch_to_write]
                    next_batch_to_write += 1
                 else:
                    print(f"⚠️ Missing batch {next_batch_to_write} at the end.")
                    break

        finally:
            out.release()

        total_time = time.time() - self.start_time
        self.print_final_statistics(total_time)
        print(f"🎉 Translation complete! Output: {self.output_path}")

    def _write_batch_frames(self, out: cv2.VideoWriter, frames: List[Tuple[int, np.ndarray]]):
        """Writes a batch of frames to the video file and updates stats."""
        frames.sort(key=lambda x: x[0])
        for _, processed_frame in frames:
            if processed_frame is not None:
                out.write(processed_frame)
        
        with self.progress_lock:
            self.processing_stats['total_frames_written'] += len(frames)

    def print_final_statistics(self, total_time: float):
        """Print comprehensive processing statistics."""
        stats = self.processing_stats
        print("\n" + "="*70)
        print("🎯 HIGH-PERFORMANCE PROCESSING STATISTICS")
        print("="*70)
        print(f"⏱️  Total processing time:    {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"🎬 Total frames written:     {stats['total_frames_written']:,}")
        print(f"📦 Batches processed:        {stats['batches_processed']:,}")
        print(f"📏 Batch size:               {self.batch_size}")
        print(f"🔧 CPU worker threads:       {self.max_workers}")
        print(f"💡 GPU concurrency limit:    {self.gpu_concurrency_limit}")
        print(f"💾 Memory limit (monitor):   {self.memory_limit_gb}GB")
        print(f"📈 Peak memory usage:        {stats['peak_memory_gb']:.2f}GB")
        print(f"⚠️  Memory warnings:          {stats['memory_warnings']}")

        total_translations = stats['cache_hits'] + stats['cache_misses']
        if total_translations > 0:
            cache_hit_rate = (stats['cache_hits'] / total_translations) * 100
            print(f"\n🔤 TRANSLATION CACHE STATISTICS:")
            print(f"✅ Cache hits:               {stats['cache_hits']:,}")
            print(f"❌ Cache misses:             {stats['cache_misses']:,}")
            print(f"📊 Cache hit rate:           {cache_hit_rate:.1f}%")

        if total_time > 0 and stats['total_frames_written'] > 0:
            fps_processed = stats['total_frames_written'] / total_time
            realtime_factor = fps_processed / self.fps if self.fps > 0 else 0
            print(f"\n🚀 PERFORMANCE METRICS:")
            print(f"⚡ Processing FPS:           {fps_processed:.2f}")
            print(f"📺 Original video FPS:       {self.fps:.2f}")
            print(f"🏃 Realtime factor:          {realtime_factor:.2f}x")
            if realtime_factor >= 1.0: print("✅ Processing faster than realtime!")
        print("="*70)

if __name__ == "__main__":
    # --- CHOOSE YOUR CONFIGURATION ---

    # Optimal configuration for a high-end system like RTX 4050 + Ryzen 9 7945HS + 32GB RAM
    high_performance_config = {
        'batch_size': 200,
        'max_workers': 20,
        'memory_limit_gb': 28,
        'gpu_concurrency_limit': 3,
        'difference_threshold': 0.08,
        'max_consecutive_skips': 100
    }

    # A more balanced configuration for mid-range systems
    balanced_config = {
        'batch_size': 100,
        'max_workers': 10,
        'memory_limit_gb': 12,
        'gpu_concurrency_limit': 2,
        'difference_threshold': 0.07
    }
    
    # Select the configuration profile to use for the run
    config_to_use = high_performance_config

    # --- RUN THE PIPELINE ---
    try:
        pipeline = MemoryOptimizedPipeline(
            video_path="test_video.mp4", # <-- IMPORTANT: Make sure this path is correct
            source_lang='fr',
            dest_lang='en',
            output_path='translated_video_fast.mp4',
            **config_to_use
        )
        print(f"\n🎯 Starting processing with config: {config_to_use}\n")
        pipeline.process_video()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Please ensure the 'video_path' in the '__main__' block points to a valid video file.")
    except ImportError as e:
        print(f"\nERROR: A required module is missing: {e}")
        print("Please ensure all required modules (ocr, translator, frames) are installed and accessible.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

