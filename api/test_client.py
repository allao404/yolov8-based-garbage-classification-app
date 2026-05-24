"""
Test client for Garbage Classification API
Example usage of the API endpoints
垃圾分类API测试客户端，演示各接口调用方法
"""

# 导入依赖库
import requests   # 用于发送HTTP请求
import json       # 用于处理JSON数据
from pathlib import Path  # 用于处理文件路径
import argparse   # 用于解析命令行参数


def test_api_health(base_url="http://localhost:8000"):
    """
    测试API健康检查接口
    :param base_url: API服务的基础URL，默认本地地址
    :return: API是否健康（True/False）
    """
    print("\n" + "="*60)
    print("Testing API Health Check")
    print("="*60)

    try:
        # 发送GET请求到健康检查接口
        response = requests.get(f"{base_url}/health")
        response.raise_for_status()  # 如果HTTP状态码不是200，抛出异常

        # 解析返回的JSON数据
        data = response.json()
        print(json.dumps(data, indent=2))  # 格式化打印返回结果

        # 检查API状态和模型加载情况
        if data.get('status') == 'healthy' and data.get('model_loaded'):
            print("\n✓ API is healthy and ready!")
            return True
        else:
            print("\n✗ API is not ready")
            return False

    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def test_detect_trash(image_path, base_url="http://localhost:8000"):
    """
    测试垃圾检测接口，上传图片并获取检测结果
    :param image_path: 待检测图片的路径
    :param base_url: API服务的基础URL
    :return: 检测结果的JSON数据（失败时返回None）
    """
    print("\n" + "="*60)
    print(f"Testing Trash Detection")
    print(f"Image: {image_path}")
    print("="*60)

    image_path = Path(image_path)  # 转换为Path对象，方便路径操作

    # 检查图片文件是否存在
    if not image_path.exists():
        print(f"\n✗ Error: Image not found: {image_path}")
        return None

    try:
        # 准备文件上传
        with open(image_path, 'rb') as f:
            # 构造文件上传所需的参数：文件名、文件对象、MIME类型
            files = {'image': (image_path.name, f, 'image/jpeg')}

            # 发送POST请求到检测接口
            print("\nSending request...")
            response = requests.post(
                f"{base_url}/v1/detect_trash",
                files=files
            )
            response.raise_for_status()  # 检查请求是否成功

        # 解析返回的JSON结果
        data = response.json()

        # 打印基础信息
        print(f"\nStatus: {data['status']}")
        print(f"Detection Count: {data['detection_count']}")
        print(f"Inference Time: {data['inference_time_ms']:.2f} ms")

        # 如果检测到了目标，打印详细信息
        if data['detection_count'] > 0:
            print(f"\nDetections:")
            print("-" * 60)

            # 遍历每个检测结果
            for i, detection in enumerate(data['detections'], 1):
                print(f"\nObject {i}:")
                print(f"  Specific Name: {detection['specific_name']}")  # 细分类别
                print(f"  General Category: {detection['general_category']}")  # 大类
                print(f"  Confidence: {detection['confidence']:.2%}")  # 置信度（百分比格式）
                print(f"  BBox (xyxy): {detection['bbox_xyxy']}")  # 检测框坐标

        else:
            print("\nNo objects detected in the image.")

        print("\n✓ Detection completed successfully!")
        return data

    except requests.exceptions.RequestException as e:
        print(f"\n✗ Request Error: {e}")
        # 如果有响应内容，打印出来方便排查
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return None

    except Exception as e:
        print(f"\n✗ Error: {e}")
        return None


def test_get_categories(base_url="http://localhost:8000"):
    """
    测试获取支持的垃圾类别接口
    :param base_url: API服务的基础URL
    :return: 类别信息的JSON数据（失败时返回None）
    """
    print("\n" + "="*60)
    print("Testing Get Categories")
    print("="*60)

    try:
        # 发送GET请求获取类别信息
        response = requests.get(f"{base_url}/v1/categories")
        response.raise_for_status()

        data = response.json()

        # 打印类别统计信息
        print(f"\nTotal Classes: {data['total_classes']}")
        print(f"\nGeneral Categories:")
        for category, count in data['general_categories'].items():
            print(f"  {category}: {count} classes")

        print("\n✓ Categories retrieved successfully!")
        return data

    except Exception as e:
        print(f"\n✗ Error: {e}")
        return None


def main():
    """主函数，解析命令行参数并执行测试"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="Test client for Garbage Classification API"
    )
    # 添加参数：--image 图片路径
    parser.add_argument(
        '--image',
        type=str,
        help='Path to test image'
    )
    # 添加参数：--url API服务地址
    parser.add_argument(
        '--url',
        type=str,
        default='http://localhost:8000',
        help='API base URL (default: http://localhost:8000)'
    )
    # 添加参数：--all 运行所有测试
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all tests'
    )

    args = parser.parse_args()  # 解析参数

    print("\n" + "="*60)
    print("Garbage Classification API - Test Client")
    print("="*60)

    # 第一步：测试API健康状态
    healthy = test_api_health(args.url)

    if not healthy:
        print("\n⚠ Warning: API is not healthy. Some tests may fail.")

    # 第二步：测试获取类别信息接口（--all 或 没指定图片时执行）
    if args.all or not args.image:
        test_get_categories(args.url)

    # 第三步：测试垃圾检测接口（如果提供了图片路径）
    if args.image:
        test_detect_trash(args.image, args.url)
    elif args.all:
        print("\n⚠ No test image provided. Skipping detection test.")
        print("  Use --image <path> to test detection.")

    print("\n" + "="*60)
    print("Tests completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
