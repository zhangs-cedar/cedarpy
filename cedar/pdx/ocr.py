from paddleocr import PaddleOCR
from cedar.image import imread
from zhconv import convert


def extract_strings(data):
    result = []
    if isinstance(data, list):
        for item in data:
            result.extend(extract_strings(item))
    elif isinstance(data, tuple):
        for item in data:
            result.extend(extract_strings(item))
    elif isinstance(data, str):
        result.append(data)
    new_result = "".join([convert(r, "zh-tw") for r in result])
    return new_result


class OCR:
    def __init__(self):
        self.pocr = PaddleOCR(lang="ch", show_log=False)  # need to run only once to download and load model into memory

    def set_shape(self, img_path):
        img_cv2 = imread(img_path)
        h, w = img_cv2.shape[:2]
        print("image shape is w: {}, h: {}".format(w, h))

    def __call__(self, img):
        res = ""
        try:
            result = self.pocr.ocr(img, cls=False)
            res = extract_strings(result)
        except:
            print(img)
        return res


if __name__ == "__main__":
    img_path = "/Users/zhangsong/workspace/OpenSource/create_idea/old/frames/frame_7.png"
    ocr = OCR()
    ocr.set_shape(img_path)
    result = ocr(img_path)
    print(result)
