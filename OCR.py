import cv2
import numpy as np
import mss
import pytesseract
import tts

# py "D:\OCR.py"

def ocr():
    with mss.mss() as sct:
        monitors = sct.monitors[1]

        height = monitors['height']
        part_width = monitors['width'] // 3
        screen = {"top": 100, "left": part_width * 2, "width": part_width, "height": (height)}

        img = np.array(sct.grab(screen))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        return frame

def ocr_screenshot_to_text():
    with mss.mss() as sct:
        monitors = sct.monitors[1]

        height = monitors['height']
        part_width = monitors['width'] // 3
        screen = {"top": 100, "left": part_width * 2, "width": part_width, "height": (height)}

        last_text = ""

        while True:
            img = np.array(sct.grab(screen))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

            text = pytesseract.image_to_string(gray, lang="eng", config="--psm 6").strip()

            if text and text != last_text:      
                last_text = text
                print(text, "\n")
                tts.speak(text)
