"""
基础特征提取模块

提供多尺度特征提取功能，包括强度、边缘和纹理特征。
"""

import itertools
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import combinations_with_replacement
from typing import List, Tuple, Optional, Callable, Any

from skimage import feature, filters
from skimage.util.dtype import img_as_float32


def _texture_filter(gaussian_filtered: np.ndarray) -> Tuple[np.ndarray, ...]:
    """计算纹理特征

    Args:
        gaussian_filtered: 高斯滤波后的图像

    Returns:
        Tuple[np.ndarray, ...]: 纹理特征值
    """
    H_elems = [
        np.gradient(np.gradient(gaussian_filtered)[ax0], axis=ax1)
        for ax0, ax1 in combinations_with_replacement(range(gaussian_filtered.ndim), 2)
    ]
    eigvals = feature.hessian_matrix_eigvals(H_elems)
    return eigvals


def _singlescale_basic_features_singlechannel(
    img: np.ndarray,
    sigma: float,
    intensity: bool = True,
    edges: bool = True,
    texture: bool = True,
) -> Tuple[np.ndarray, ...]:
    """单尺度单通道基础特征提取

    Args:
        img: 输入图像
        sigma: 高斯核标准差
        intensity: 是否计算强度特征
        edges: 是否计算边缘特征
        texture: 是否计算纹理特征

    Returns:
        Tuple[np.ndarray, ...]: 提取的特征
    """
    results = ()
    gaussian_filtered = filters.gaussian(img, sigma, preserve_range=False)

    if intensity:
        results += (gaussian_filtered,)
    if edges:
        results += (filters.sobel(gaussian_filtered),)
    if texture:
        results += (*_texture_filter(gaussian_filtered),)

    return results


def _multiscale_basic_features_singlechannel(
    img: np.ndarray,
    intensity: bool = True,
    edges: bool = True,
    texture: bool = True,
    sigma_min: float = 0.5,
    sigma_max: float = 16,
    num_sigma: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> itertools.chain:
    """多尺度单通道基础特征提取

    Args:
        img: 输入图像，可以是灰度或多通道
        intensity: 是否计算强度特征，默认为True
        edges: 是否计算边缘特征，默认为True
        texture: 是否计算纹理特征，默认为True
        sigma_min: 高斯核最小值，默认为0.5
        sigma_max: 高斯核最大值，默认为16
        num_sigma: 高斯核数量，如果为None则自动计算
        num_workers: 并行线程数，如果为None则使用所有可用核心

    Returns:
        itertools.chain: 特征列表的迭代器
    """
    # 转换为float32以提高计算速度
    img = np.ascontiguousarray(img_as_float32(img))

    if num_sigma is None:
        num_sigma = int(np.log2(sigma_max) - np.log2(sigma_min) + 1)

    sigmas = np.logspace(
        np.log2(sigma_min),
        np.log2(sigma_max),
        num=num_sigma,
        base=2,
        endpoint=True,
    )

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        out_sigmas = list(
            ex.map(
                lambda s: _singlescale_basic_features_singlechannel(img, s, intensity=intensity, edges=edges, texture=texture),
                sigmas,
            )
        )

    features = itertools.chain.from_iterable(out_sigmas)
    return features


def multiscale_basic_features(
    image: np.ndarray,
    multichannel: bool = False,
    intensity: bool = True,
    edges: bool = True,
    texture: bool = True,
    sigma_min: float = 0.5,
    sigma_max: float = 16,
    num_sigma: Optional[int] = None,
    num_workers: Optional[int] = None,
    *,
    channel_axis: Optional[int] = None,
) -> np.ndarray:
    """多尺度基础特征提取

    通过高斯模糊在不同尺度下计算强度、梯度强度和局部结构特征。

    Args:
        image: 输入图像，可以是灰度或多通道
        multichannel: 是否多通道，已弃用，请使用channel_axis参数
        intensity: 是否计算强度特征，默认为True
        edges: 是否计算边缘特征，默认为True
        texture: 是否计算纹理特征，默认为True
        sigma_min: 高斯核最小值，默认为0.5
        sigma_max: 高斯核最大值，默认为16
        num_sigma: 高斯核数量，如果为None则自动计算
        num_workers: 并行线程数，如果为None则使用所有可用核心
        channel_axis: 通道轴，如果为None则假设为灰度图像

    Returns:
        np.ndarray: 特征数组，形状为 image.shape + (n_features,)

    Raises:
        ValueError: 当所有特征类型都为False时
    """
    if not any([intensity, edges, texture]):
        raise ValueError('At least one of `intensity`, `edges` or `textures` ' 'must be True for features to be computed.')

    if channel_axis is None:
        image = image[..., np.newaxis]
        channel_axis = -1
    elif channel_axis != -1:
        image = np.moveaxis(image, channel_axis, -1)

    all_results = (
        _multiscale_basic_features_singlechannel(
            image[..., dim],
            intensity=intensity,
            edges=edges,
            texture=texture,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            num_sigma=num_sigma,
            num_workers=num_workers,
        )
        for dim in range(image.shape[-1])
    )

    features = list(itertools.chain.from_iterable(all_results))
    out = np.stack(features, axis=-1)
    return out


def multiscale_features(
    intensity: bool = True,
    edges: bool = True,
    texture: bool = True,
    sigma_min: int = 1,
    sigma_max: int = 3,
    num_workers: Optional[int] = None,
    channel_axis: int = -1,
) -> Callable[[np.ndarray], np.ndarray]:
    """创建多尺度特征提取函数

    Args:
        intensity: 是否计算灰度值，默认为True
        edges: 是否计算边缘，默认为True
        texture: 是否计算纹理，默认为True
        sigma_min: 最小的高斯核大小，默认为1
        sigma_max: 最大的高斯核大小，默认为3
        num_workers: 并行线程数，默认为None
        channel_axis: 通道轴，默认为-1

    Returns:
        Callable[[np.ndarray], np.ndarray]: 多尺度特征提取函数
    """
    features_func = partial(
        multiscale_basic_features,
        intensity=intensity,
        edges=edges,
        texture=texture,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        num_workers=num_workers,
        channel_axis=channel_axis,
    )
    return features_func
