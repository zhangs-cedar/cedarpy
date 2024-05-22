import io
import time
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image


class ModelClient:
    def __init__(self, server_url):
        self.server_url = server_url

    def _prepare_image(self, image):
        if isinstance(image, str):  # 图片路径
            image = Image.open(image)
        elif isinstance(image, np.ndarray):  # np矩阵
            image = Image.fromarray(image)
        elif isinstance(image, plt.Figure):  # plt读取的对象
            image_byte_array = io.BytesIO()
            image.savefig(image_byte_array, format="JPEG")
            image = Image.open(image_byte_array)
        else:
            raise ValueError("Unsupported image type")

        image_byte_array = io.BytesIO()
        image.save(image_byte_array, format="JPEG")
        return image_byte_array.getvalue()

    def predict(self, images):
        """
        Args:
            images: 图片路径或者 np 矩阵或者 plt 读取的对象，可以是单个图像或图像列表
        Returns:
            dict: 预测结果
        """
        if not isinstance(images, list):
            images = [images]
        print(images)
        files = {}
        for index,image_file in enumerate(images):
            files[str(index)+".jpg"] = (str(index)+".jpg", self._prepare_image(image_file))
        response = requests.post(f"{self.server_url}/predict", files=files)
        if response.status_code == 200:
            return eval(response.json())
        else:
            return {"error": response.text}

    def shutdown(self):
        try:
            requests.get(self.server_url + "/shutdown")
        except requests.exceptions.ConnectionError:
            print("Server has been shutdown")


if __name__ == "__main__":
    server_url = "http://127.0.0.1:24401"  # 请修改为你的服务器地址,不要使用localhost 速度会很慢,因为localhost会使用ipv6
    client = ModelClient(server_url)

    image_path = r"C:\home\workspace\datasets\D0001\JPEGImages\0001.jpg"
    start_time = time.time()
    result = client.predict([image_path, image_path])
    print("Inference time: {:.4f} seconds".format(time.time() - start_time))
    print("Prediction result:", result)
    # res = client.shutdown()
    # print(res)

    # 示例：如果要用 np 矩阵进行预测
    image_matrix = np.random.rand(10, 10, 3) * 255
    image_matrix = image_matrix.astype(np.uint8)
    result = client.predict(image_matrix)

    # 示例：如果要用 plt 读取的对象进行预测
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])
    result = client.predict(fig)

    print("Prediction result:", result)
