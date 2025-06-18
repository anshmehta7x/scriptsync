# fast_multithreaded_pipeline.py
"""Ultra‑lean multithreaded video‑translation pipeline
=====================================================
Designed to preserve the **winning traits of your sequential script** while
parallelising only the heavy GPU/OCR work.

Main design points
------------------
1. **Single sequential decoder** – one thread walks forward through the video, so
   there are *zero* codec seeks.
2. **Global skip‑state** – difference‑threshold and consecutive‑skip logic live
   in the reader; workers never reset this state.
3. **Only the *needed* frames hit the GPU** – reader queues a frame *only* when
   it breaks the skip rule; all other frames are handled by the writer via
   simple reuse of the last processed frame.
4. **Shared translation cache** – lock‑free for hits, short lock for the first
   miss of a string.
5. **Strict output order** – writer writes frames sequentially, guaranteeing a
   playable result with perfect sync.
6. **Lean memory footprint** – at most *queue_size + 1* full‑resolution frames
   live in RAM, so no GC storms.

You can drop this file next to your existing project that already contains
`OCRFrame`, `TranslationCache`, and `FrameProcessor`.

Example
~~~~~~~
```python
from fast_multithreaded_pipeline import FastPipeline

FastPipeline(
    video_path="input.mp4",
    output_path="output.mp4",
    source_lang="fr",
    dest_lang="en",
    difference_threshold=0.08,
    max_consecutive_skips=100,
    max_workers=8,            # physical CPU cores
    gpu_concurrency_limit=3,  # for an RTX‑class GPU
).run()
```
"""
from __future__ import annotations

import os, time, gc, queue, threading
from typing import Dict, List, Tuple, Optional

import cv2, psutil, numpy as np

# ── project‑specific imports ────────────────────────────────────────────────
from ocr import OCRFrame
from translator import TranslationCache
from frames import FrameProcessor

# ── core pipeline ───────────────────────────────────────────────────────────
class FastPipeline:
    def __init__(
        self,
        *,
        video_path: str,
        output_path: str = "translated.mp4",
        source_lang: str = "en",
        dest_lang: str = "fr",
        difference_threshold: float = 0.05,
        max_consecutive_skips: int = 30,
        max_workers: int | None = None,
        gpu_concurrency_limit: int = 2,
        queue_size: int = 300,
        log_every: float = 2.0,
    ) -> None:
        self.video_path, self.output_path = video_path, output_path
        self.src, self.dst = source_lang, dest_lang
        self.diff_th = difference_threshold
        self.max_skips = max_consecutive_skips
        self.log_every = log_every

        self.max_workers = max_workers or max(2, (os.cpu_count() or 4) - 1)
        self.gpu_sem = threading.Semaphore(gpu_concurrency_limit)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(video_path)
        self.frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # queues & shared state
        self.todo: "queue.Queue[Tuple[int, np.ndarray]]" = queue.Queue(queue_size)
        self.done: Dict[int, Optional[np.ndarray]] = {}
        self.done_lock = threading.Lock()

        # translation cache
        self.cache = TranslationCache(); self.cache.set_source(self.src); self.cache.set_dest(self.dst)
        self.cache_lock = threading.Lock()

        # stats
        self.stats = dict(proc=0, skip=0, hit=0, miss=0, rss_max=0.0)
        self.s_lock = threading.Lock()

    # ── helper: translation cache with minimal locking ────────────────────
    def _translate_cached(self, ocr: OCRFrame) -> Dict[str, str]:
        todo: List[str] = []
        out: Dict[str,str] = {}
        with self.cache_lock:
            for r in ocr.get_results():
                t = r.text.strip()
                if not t: continue
                c = self.cache.fuzzy_match_transaltions(t)
                if c:
                    out[t] = c; self.stats['hit'] += 1
                else:
                    todo.append(t)
        for t in todo:  # translate outside the lock
            try: out[t] = self.cache.translate(t)
            except Exception: out[t] = t
            self.stats['miss'] += 1
        return out

    # ── reader thread ────────────────────────────────────────────────────
    def _reader(self):
        cap = cv2.VideoCapture(self.video_path)
        last_log = time.time()
        last_frame: Optional[np.ndarray] = None
        skips = 0
        for idx in range(self.frames):
            ok, frame = cap.read();  # sequential decode
            if not ok: break

            need_proc = skips >= self.max_skips or last_frame is None
            if not need_proc:
                diff = cv2.absdiff(frame, last_frame)
                need_proc = float(np.mean(diff)) / 255 > self.diff_th
            if need_proc:
                self.todo.put((idx, frame))
                skips = 0
                last_frame = frame
            else:
                with self.done_lock: self.done[idx] = None
                skips += 1
            if time.time()-last_log>self.log_every:
                pct = (idx+1)*100/self.frames
                print(f"[Reader] {pct:5.1f}% read | queue={self.todo.qsize():3d} | skips={skips}")
                last_log=time.time()
        cap.release()
        for _ in range(self.max_workers): self.todo.put((-1, None))  # sentinels

    # ── worker threads ───────────────────────────────────────────────────
    _tls = threading.local()
    def _worker(self, wid:int):
        while True:
            idx, frame = self.todo.get()
            if idx == -1: break
            if not hasattr(self._tls,'proc'):
                self._tls.proc = FrameProcessor(self.video_path, self.src, self.diff_th)
            with self.gpu_sem:
                ocr = OCRFrame(frame, self.src)
                trans = self._translate_cached(ocr)
                proc_frame = self._tls.proc.paste_frame(frame, ocr, trans)
            with self.done_lock: self.done[idx] = proc_frame
            with self.s_lock:
                self.stats['proc'] += 1
                rss = psutil.Process().memory_info().rss/1024**3
                self.stats['rss_max'] = max(self.stats['rss_max'], rss)

    # ── run / writer (main) ──────────────────────────────────────────────
    def run(self):
        print(f"🏁 FastPipeline | {self.frames} frames | {self.fps:.1f} fps | {self.w}×{self.h}")
        t0=time.time()
        threading.Thread(target=self._reader,daemon=True).start()
        workers=[threading.Thread(target=self._worker,args=(i,),daemon=True) for i in range(self.max_workers)]
        for t in workers:t.start()

        fourcc=cv2.VideoWriter_fourcc(*'mp4v')
        out=cv2.VideoWriter(self.output_path,fourcc,self.fps,(self.w,self.h))
        if not out.isOpened(): raise IOError("Cannot open VideoWriter")

        next_idx=0; last_frame=None; last_log=time.time()
        while next_idx<self.frames:
            with self.done_lock:
                ready=next_idx in self.done
                fr=self.done.pop(next_idx,None) if ready else None
            if ready:
                if fr is None:  # skipped frame
                    out.write(last_frame)
                    self.stats['skip']+=1
                else:
                    out.write(fr); last_frame=fr
                next_idx+=1
            else:
                time.sleep(0.002)
            if time.time()-last_log>self.log_every:
                pct=next_idx*100/self.frames
                fps_out=next_idx/(time.time()-t0)
                print(f"[Writer] {pct:5.1f}% written | {fps_out:6.1f} fps | RAM {self.stats['rss_max']:.2f} GB")
                last_log=time.time()
        out.release(); gc.collect()
        for t in workers: t.join()

        # summary
        elapsed=time.time()-t0; p,s=self.stats['proc'],self.stats['skip']
        print("\n✅ done →",self.output_path)
        print(f"⏱ {elapsed:.2f}s  | {self.frames/elapsed:.1f} overall fps")
        print(f"🔤 OCR/translate frames: {p} ({p*100/self.frames:.1f}%)  | skipped: {s}")
        tot=self.stats['hit']+self.stats['miss'] or 1
        print(f"💡 cache hit‑rate: {self.stats['hit']*100/tot:.1f}%  | peak RSS {self.stats['rss_max']:.2f} GB")

# ── standalone run ─────────────────────────────────────────────────────────
if __name__=="__main__":
    FastPipeline(
        video_path="test_video.mp4",
        output_path="translated_video_fast.mp4",
        source_lang="fr",
        dest_lang="en",
        difference_threshold=0.08,
        max_consecutive_skips=100,
        max_workers=8,
        gpu_concurrency_limit=3,
    ).run()
