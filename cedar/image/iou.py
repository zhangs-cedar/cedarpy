def calculate_iou(box1, box2):
    """
    计算两个矩形框之间的交并比(IoU)

    Args:
        box1 (tuple[float, float, float, float]): 矩形框1的坐标和尺寸，格式为(x1, y1, w1, h1)，
            其中(x1, y1)为矩形框左上角坐标，(w1, h1)为矩形框的宽和高。
        box2 (Tuple[float, float, float, float]): 矩形框2的坐标和尺寸，格式为(x2, y2, w2, h2)，
            其中(x2, y2)为矩形框左上角坐标，(w2, h2)为矩形框的宽和高。

    Returns:
        float: 两个矩形框之间的交并比(IoU)，范围为[0, 1]。

    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    # 计算交集区域的坐标
    x_intersect = max(x1, x2)
    y_intersect = max(y1, y2)
    w_intersect = min(x1 + w1, x2 + w2) - x_intersect
    h_intersect = min(y1 + h1, y2 + h2) - y_intersect
    # 计算交集区域的面积
    intersection_area = max(0, w_intersect) * max(0, h_intersect)
    # 计算并集区域的面积
    union_area = w1 * h1 + w2 * h2 - intersection_area
    # 计算 IoU
    iou = intersection_area / union_area
    return iou


def merge_boxes(boxes, iou_threshold):
    """
    合并重叠的边界框。

    Args:
        boxes (list[list[float]]): 边界框列表，每个边界框由 [x, y, width, height] 表示。
        iou_threshold (float): 重叠阈值，用于判断两个边界框是否重叠。

    Returns:
        list[list[float]]: 合并后的边界框列表，每个边界框由 [x, y, width, height] 表示。

    """
    merged_boxes = []
    # 遍历所有边界框
    for i, box in enumerate(boxes):
        # 检查当前边界框是否与其他边界框重叠
        overlapped = False
        for j, merged_box in enumerate(merged_boxes):
            if calculate_iou(box, merged_box) > iou_threshold:
                # 如果重叠，则合并边界框
                x = min(box[0], merged_box[0])
                y = min(box[1], merged_box[1])
                w = max(box[0] + box[2], merged_box[0] + merged_box[2]) - x
                h = max(box[1] + box[3], merged_box[1] + merged_box[3]) - y
                merged_boxes[j] = [x, y, w, h]
                overlapped = True
                break
        # 如果当前边界框没有与任何已合并的边界框重叠，则将其添加到合并边界框列表中
        if not overlapped:
            merged_boxes.append(box)

    return merged_boxes
