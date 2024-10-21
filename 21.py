import cv2
import numpy as np
import difflib
import json
from paddleocr import PaddleOCR

reference_numbers = ["61S012", "DIN62FE", "DE 31 2G SMAW"]

ocr_engine = PaddleOCR(
    det_model_dir="ocr_models/ch_PP-OCRv4_det_infer",
    rec_model_dir="ocr_models/ch_PP-OCRv4_rec_infer",
    cls_model_dir="ocr_models/ch_ppocr_mobile_v2.0_cls_infer",
    rec_char_dict_path="ocr_models/ppocr_keys_v1.txt",
    use_angle_cls=False,
    lang="en",
    det_db_box_thresh=0.3,
    drop_score=0.2
)

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return processed_img

def check_flip_confidence(image):
    projection = np.sum(image, axis=1)
    smooth_projection = np.convolve(projection, np.ones(20)/20, mode='same')
    peaks_in_lower_half = np.sum(smooth_projection[:len(smooth_projection)//2] > smooth_projection.mean())
    flip_confidence = (peaks_in_lower_half / len(projection)) * 100
    flip_status = "flipped" if flip_confidence < 30 else "unflipped" if flip_confidence > 70 else "unknown"
    return flip_status


def best_match(text, ref_numbers):
    matches = difflib.get_close_matches(text, ref_numbers, n=1, cutoff=0.5)
    return matches[0] if matches else None

def ocr_text(image):
    ocr_result = ocr_engine.ocr(image, cls=False)
    return ocr_result[0][0][1][0].strip() if ocr_result and ocr_result[0] else ""

def process_image(image_path):
    image = cv2.imread(image_path)
    preprocessed_image = preprocess_image(image)
    original_text = ocr_text(preprocessed_image)
    flipped_image = cv2.flip(preprocessed_image, 0)
    flipped_text = ocr_text(flipped_image)
    flip_status = check_flip_confidence(preprocessed_image)
    best_ref_number = best_match(original_text if flip_status != "flipped" else flipped_text, reference_numbers)
    result = {
        "flip_status": flip_status,
        "best_match_reference": best_ref_number if best_ref_number else "No match",
        "texts": {
            "original_text": original_text,
            "flipped_text": flipped_text
        }
    }
    return result

image_path = r"C:\Users\sruth\Desktop\bounded_after_prep\filtered_images\merged_region_0.png"
result = process_image(image_path)
print(json.dumps(result, indent=4))

