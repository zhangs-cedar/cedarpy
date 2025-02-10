from paddleocr import PaddleOCR
from cedar.image import imread
from zhconv import convert

class OCR:
    def __init__(self):
        self.pocr = PaddleOCR(lang="ch",show_log=False)  # need to run only once to download and load model into memory

    def set_shape(self, img_path):
        img_cv2 = imread(img_path)
        h, w = img_cv2.shape[:2]
        print("image shape is w: {}, h: {}".format(w, h))

    def __call__(self, img):
        pos, center_coordinates, res = [], [], ""
        try:
            result = self.pocr.ocr(img, cls=False)
            pos, res = result[0][0][0], result[0][0][1][0]
            # 计算中心坐标
            center_x = sum(x for x, y in pos) / len(pos)
            center_y = sum(y for x, y in pos) / len(pos)
            center_coordinates = [center_x, center_y]
        except:
            print(img)
        simplified_text = convert(res, 'zh-cn')
        return pos, center_coordinates, simplified_text


if __name__ == "__main__":
    img_path = "/Users/zhangsong/workspace/OpenSource/create_idea/frames/frame_0.png"
    ocr = OCR()
    ocr.set_shape(img_path)
    result = ocr(img_path)
    print(result)
