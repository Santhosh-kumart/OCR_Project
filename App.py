import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import re
import cv2
from ultralytics import YOLO
import easyocr

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Vehicle Lookup System", page_icon="🚗")
st.title("🚗 Vehicle Lookup System (YOLO + EasyOCR)")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("Book1.csv", dtype={"CarNumber": str})

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    yolo = YOLO("best.pt")
    reader = easyocr.Reader(['en'], gpu=False)
    return yolo, reader

yolo, ocr = load_models()

# Only allow plate-relevant characters so EasyOCR doesn't hallucinate symbols
OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# -----------------------------
# CHARACTER CONFUSION MAPS
# -----------------------------
# NOTE: kept symmetric on purpose -- if a letter can be misread as a digit,
# the digit can just as easily be misread as that letter (e.g. Z <-> 2).
LETTER_TO_DIGIT = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "Q": "0"}
DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"}


def _normalize_segment(seg, to_digit):
    """Apply OCR confusion corrections only within a specific segment
    (e.g. only where digits are expected, or only where letters are expected)."""
    mapping = LETTER_TO_DIGIT if to_digit else DIGIT_TO_LETTER
    return "".join(mapping.get(ch, ch) for ch in seg)


# -----------------------------
# VEHICLE NUMBER EXTRACTION
# -----------------------------
def extract_tn_number(text):
    """
    Extract a TN-format plate: TN + 2 digits + 1-2 letters + 4 digits.
    Fixes common OCR confusions (O/0, I/1, S/5, B/8, Z/2 ...) but ONLY within
    the segment where a digit or letter is actually expected, so real letters
    like 'B' or 'S' inside the RTO code aren't wrongly turned into digits.
    """
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)  # strip spaces, dashes, punctuation

    # 1) Try a direct match first (best case: OCR was already clean)
    match = re.search(r"TN\d{2}[A-Z]{1,2}\d{4}", text)
    if match:
        return match.group()

    def find_candidates(prefix):
        """Locate `prefix` in the text and try to reconstruct a TN plate
        after it. The returned plate always starts with the canonical 'TN'
        regardless of what the OCR prefix actually was -- we already know
        the vehicle is TN-registered, we're just correcting the OCR read."""
        idx = text.find(prefix)
        if idx == -1:
            return []

        rest = text[idx + len(prefix):]
        results = []

        # Try both 1-letter and 2-letter RTO codes and keep every candidate
        # that parses into a valid plate. We used to return on the FIRST
        # match, which meant a valid-but-wrong letter_len=1 parse could win
        # even when it threw away a trailing character that letter_len=2
        # would have used. Instead, collect all valid candidates and prefer
        # whichever consumes the OCR text most fully (least leftover) --
        # silently dropping characters is a worse failure mode than a
        # slightly ambiguous parse.
        for letter_len in (1, 2):
            total_len = 2 + letter_len + 4
            if len(rest) < total_len:
                continue

            candidate = rest[:total_len]
            state_digits = _normalize_segment(candidate[0:2], to_digit=True)
            letters = _normalize_segment(candidate[2:2 + letter_len], to_digit=False)
            number_digits = _normalize_segment(candidate[2 + letter_len:2 + letter_len + 4], to_digit=True)

            plate = f"TN{state_digits}{letters}{number_digits}"
            if re.fullmatch(r"TN\d{2}[A-Z]{1,2}\d{4}", plate):
                leftover = len(rest) - total_len
                results.append((leftover, plate))

        return results

    # 2) Fallback: locate the literal "TN" prefix and reconstruct from there.
    candidates = find_candidates("TN")

    # 3) If that failed, the OCR may have misread the *prefix* itself, not
    # just the digits/letters after it. "TN" is commonly misread as "TM"
    # (N and M both have tall vertical strokes) or "TH"/"IN"/"IM" at low
    # resolution. Only tried as a second pass, so we don't start guessing
    # wildly when a clean "TN" match already exists.
    if not candidates:
        for alt_prefix in ("TM", "TH", "IN", "IM", "TW"):
            candidates = find_candidates(alt_prefix)
            if candidates:
                break

    if not candidates:
        return None

    candidates.sort(key=lambda c: c[0])  # least leftover first
    return candidates[0][1]


# -----------------------------
# PLATE PREPROCESSING
# -----------------------------
def preprocess_variants(plate_img):
    """
    Return several differently-preprocessed versions of the plate crop.
    A single global-Otsu threshold (the old approach) can merge adjacent
    characters into one blob under uneven lighting, which is what caused the
    whole plate to come back as a single low-confidence OCR detection.
    Trying multiple strategies and keeping whichever segments best fixes that.
    """
    plate = cv2.cvtColor(plate_img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    variants = []

    # Variant A: CLAHE (local contrast) -- best for uneven/glare lighting,
    # keeps grayscale (no hard threshold), often segments characters better.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.resize(enhanced, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    variants.append(("clahe", enhanced))

    # Variant B: adaptive threshold -- handles uneven illumination much
    # better than a single global Otsu threshold across the whole plate.
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    adaptive = cv2.resize(adaptive, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    variants.append(("adaptive", adaptive))

    # Variant C: original bilateral + Otsu approach, kept as a fallback,
    # but upscaled more (3x instead of 2x) to give EasyOCR more pixels per
    # character to work with.
    den = cv2.bilateralFilter(gray, 11, 17, 17)
    otsu = cv2.threshold(den, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    otsu = cv2.resize(otsu, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    variants.append(("otsu", otsu))

    return variants


def run_ocr_best(plate_img, ocr_reader):
    """
    Run OCR on every preprocessing variant and keep the best result.
    'Best' = a result that already parses into a valid TN plate, otherwise
    the one with the highest OCR confidence.
    """
    best = None  # (score, label, text_list, confidence, raw_result)

    for label, processed in preprocess_variants(plate_img):
        result = ocr_reader.readtext(processed, allowlist=OCR_ALLOWLIST)
        if not result:
            continue

        result_sorted = sorted(result, key=lambda d: d[0][0][0])
        text_list = [d[1] for d in result_sorted]
        confidence = max(d[2] for d in result_sorted)
        joined = "".join(text_list)

        parses_clean = extract_tn_number(joined) is not None
        # Rank: a clean parse always beats a non-parse; ties broken by confidence.
        score = (1 if parses_clean else 0, confidence)

        if best is None or score > best[0]:
            best = (score, label, text_list, confidence, result)

    if best is None:
        return None
    _, label, text_list, confidence, raw_result = best
    return {
        "variant": label,
        "text_list": text_list,
        "confidence": confidence,
        "raw_result": raw_result,
        "joined": "".join(text_list),
    }


# -----------------------------
# CAMERA INPUT
# -----------------------------
image_file = st.camera_input("📸 Take Vehicle Photo")

if image_file:

    image = Image.open(image_file).convert("RGB")
    img = np.array(image)

    st.image(image, caption="Captured Image", use_container_width=True)

    with st.spinner("🔍 Detecting Number Plate..."):

        results = yolo(img)

        plate_found = False
        plate_text_final = ""
        best_confidence = 0.0
        best_variant = None

        for r in results:

            if len(r.boxes) == 0:
                continue

            # Highest confidence box
            best_box = max(r.boxes, key=lambda b: float(b.conf[0]))

            x1, y1, x2, y2 = map(int, best_box.xyxy[0])

            h, w = img.shape[:2]

            # Add a small padding margin around the box -- a tight crop can
            # clip the edges of the first/last character, which also
            # encourages EasyOCR to merge remaining characters into one blob.
            pad_x = int((x2 - x1) * 0.05)
            pad_y = int((y2 - y1) * 0.15)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)

            # Draw rectangle
            display = img.copy()
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            st.image(display, caption="Detected Number Plate", use_container_width=True)

            plate_img = img[y1:y2, x1:x2]

            if plate_img.size == 0:
                continue

            st.image(plate_img, caption="Plate Crop", use_container_width=True)

            # -----------------------------
            # OCR (tries multiple preprocessing strategies, keeps the best)
            # -----------------------------
            ocr_best = run_ocr_best(plate_img, ocr)

            if ocr_best is None:
                continue

            st.write(f"OCR variant used: **{ocr_best['variant']}**")
            st.write("OCR Raw Result:")
            st.write(ocr_best["raw_result"])

            plate_text_final = ocr_best["joined"]
            best_confidence = ocr_best["confidence"]
            best_variant = ocr_best["variant"]
            plate_found = True
            break

    if not plate_found:

        st.error("❌ No Number Plate Detected")

    else:

        st.subheader("📄 OCR Output")
        st.write(plate_text_final)

        if best_confidence < 0.30:
            st.warning(
                f"⚠ Low OCR confidence ({best_confidence:.2f}) using the '{best_variant}' variant. "
                "Result may be inaccurate — try a clearer, well-lit, closer photo."
            )

        vehicle_number = extract_tn_number(plate_text_final)

        # -----------------------------
        # MANUAL CONFIRM STEP
        # -----------------------------
        # OCR sometimes inserts or drops a stray character (e.g. reads
        # "TN09DB3529" as "TN09DBJ3529"). When that happens, several
        # different plate numbers can look equally valid to an algorithm,
        # so we don't silently guess — we show the best automatic guess
        # (or the raw text if no guess was found) and let the operator
        # confirm/correct it against the photo before looking it up.
        st.subheader("✏️ Confirm Vehicle Number")

        if vehicle_number:
            st.info(f"Best automatic guess: **{vehicle_number}**")
            default_value = vehicle_number
        else:
            st.warning(
                "❌ Could not confidently parse a TN vehicle number from the OCR text. "
                "Please check the plate photo above and correct the text below."
            )
            default_value = plate_text_final

        confirmed_number = st.text_input(
            "Vehicle number (edit if incorrect, then confirm)",
            value=default_value,
        ).strip().upper()

        if st.button("🔍 Look Up Vehicle"):

            result = df[df["CarNumber"].str.upper() == confirmed_number]

            if not result.empty:
                st.subheader("🚗 Vehicle Details")
                st.dataframe(result, use_container_width=True)
            else:
                st.warning("⚠ Vehicle not found in database")