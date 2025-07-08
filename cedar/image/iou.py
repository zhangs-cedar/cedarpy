from typing import List, Tuple


def calculate_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """计算两个矩形框之间的交并比(IoU)

    Args:
        box1: 矩形框1的坐标和尺寸，格式为(x1, y1, w1, h1)，
            其中(x1, y1)为矩形框左上角坐标，(w1, h1)为矩形框的宽和高
        box2: 矩形框2的坐标和尺寸，格式为(x2, y2, w2, h2)，
            其中(x2, y2)为矩形框左上角坐标，(w2, h2)为矩形框的宽和高

    Returns:
        float: 两个矩形框之间的交并比(IoU)，范围为[0, 1]

    Raises:
        ValueError: 当输入参数格式不正确时
    """
    if len(box1) != 4 or len(box2) != 4:
        raise ValueError('边界框必须包含4个元素: (x, y, width, height)')

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
    if union_area == 0:
        return 0.0

    iou = intersection_area / union_area
    return iou


def merge_boxes(boxes: List[List[float]], iou_threshold: float) -> List[List[float]]:
    """合并重叠的边界框

    Args:
        boxes: 边界框列表，每个边界框由 [x, y, width, height] 表示
        iou_threshold: 重叠阈值，用于判断两个边界框是否重叠

    Returns:
        List[List[float]]: 合并后的边界框列表，每个边界框由 [x, y, width, height] 表示

    Raises:
        ValueError: 当输入参数格式不正确时
    """
    if not boxes:
        return []

    if iou_threshold < 0 or iou_threshold > 1:
        raise ValueError('IoU阈值必须在[0, 1]范围内')

    merged_boxes = []

    # 遍历所有边界框
    for i, box in enumerate(boxes):
        if len(box) != 4:
            raise ValueError(f'边界框 {i} 必须包含4个元素: [x, y, width, height]')

        # 检查当前边界框是否与其他边界框重叠
        overlapped = False
        for j, merged_box in enumerate(merged_boxes):
            if calculate_iou(tuple(box), tuple(merged_box)) > iou_threshold:
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
