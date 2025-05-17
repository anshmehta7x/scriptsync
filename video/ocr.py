import easyocr
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

class OCRResult:
    def __init__(self, text: str, bbox: List[List[float]], confidence: float):
        self.text = text
        self.bbox = bbox
        self.confidence = confidence
        self.center = self._calculate_center()

    def _calculate_center(self) -> Tuple[float, float]:
        x_coords = [point[0] for point in self.bbox]
        y_coords = [point[1] for point in self.bbox]

        center_x = sum(x_coords) / len(self.bbox)
        center_y = sum(y_coords) / len(self.bbox)

        return (center_x, center_y)

    def __str__(self) -> str:
        return f"Text: '{self.text}', Confidence: {self.confidence:.2f}, Center: {self.center}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'bbox': self.bbox,
            'confidence': self.confidence,
            'center': self.center
        }

class OCRFrame:
    def __init__(self, frame, source_lang='en'):
        self.frame = frame
        self.source_lang = source_lang
        self.results: List[OCRResult] = []
        self.full_text = ""

        self._process_ocr()

    def _process_ocr(self) -> None:
        reader = easyocr.Reader([self.source_lang])
        detections = reader.readtext(self.frame)

        texts = []
        for detection in detections:
            bbox, text, confidence = detection
            ocr_result = OCRResult(text, bbox, confidence)
            self.results.append(ocr_result)
            texts.append(text)

        self.full_text = " ".join(texts)

    def get_text(self) -> str:
        return self.full_text

    def get_results(self) -> List[OCRResult]:
        return self.results

    def get_results_by_confidence(self, threshold: float = 0.5) -> List[OCRResult]:
        return [r for r in self.results if r.confidence >= threshold]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'full_text': self.full_text,
            'text_locations': {r.text: r.to_dict() for r in self.results}
        }

def ocr_on_frame(frame, source='en') -> Dict[str, Any]:
    ocr_frame = OCRFrame(frame, source)
    return ocr_frame.to_dict()

if __name__ == "__main__":
    image_path = "test_image.png"
    frame = cv2.imread(image_path)

    ocr_frame = OCRFrame(frame)
    print(f"Full Text: {ocr_frame.get_text()}")
    print(f"Found {len(ocr_frame.get_results())} text regions")

    high_conf_results = ocr_frame.get_results_by_confidence(0.7)
    print(f"\nHigh confidence results ({len(high_conf_results)}):")
    for result in high_conf_results:
        print(result)
