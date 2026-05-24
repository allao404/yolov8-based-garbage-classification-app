"""
垃圾分类API网络诊断工具

本脚本用于诊断安卓模拟器与垃圾分类API服务器之间的网络连接问题，
包含端口检测、服务连通性、网络配置、防火墙、接口功能测试等全套诊断流程。
"""

import socket
import requests
import subprocess
import sys
from pathlib import Path

def print_section(title):
    """
    打印诊断模块标题，格式化输出
    :param title: 模块标题
    """
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def check_port_listening(port=8000):
    """
    检查本地8000端口是否处于监听状态（判断API服务是否启动）
    :param port: 监听端口，默认8000
    :return: 端口监听成功返回True，失败返回False
    """
    print_section("1. 检查端口监听状态")

    try:
        # 创建TCP套接字，测试本地端口连通性
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)  # 设置超时时间1秒
        # connect_ex：连接成功返回0，失败返回错误码
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()

        if result == 0:
            print(f"✅ 端口 {port} 正在监听")
            return True
        else:
            print(f"❌ 端口 {port} 未监听")
            print(f"   请确认API服务已启动：python api/main.py")
            return False
    except Exception as e:
        print(f"❌ 检查端口时发生错误：{e}")
        return False

def check_localhost_api():
    """
    测试本地API服务健康检查接口，验证服务运行状态
    :return: 接口正常返回True，失败返回False
    """
    print_section("2. 检查本地API服务健康状态")

    try:
        # 请求健康检查接口，超时时间5秒
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API服务运行正常")
            print(f"   - 模型加载状态：{data.get('model_loaded')}")
            print(f"   - GPU可用状态：{data.get('gpu_available')}")
            print(f"   - 运行设备：{data.get('device_in_use')}")
            return True
        else:
            print(f"❌ API返回异常状态码：{response.status_code}")
            return False
    except requests.exceptions.Timeout:
        print(f"❌ API请求超时")
        print(f"   API服务可能正在加载模型，请等待后重试")
        return False
    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接到API服务器")
        print(f"   请确认API服务已启动")
        return False
    except Exception as e:
        print(f"❌ 检查API时发生错误：{e}")
        return False

def check_network_interfaces():
    """
    检查本机网络信息，提供安卓模拟器/局域网连接IP配置
    :return: 获取成功返回True，失败返回False
    """
    print_section("3. 检查网络接口信息")

    try:
        # 获取本机计算机名和局域网IP
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)

        print(f"✅ 计算机名：{hostname}")
        print(f"✅ 局域网IP：{ip_address}")
        # 关键配置提示：安卓模拟器访问本地服务固定使用10.0.2.2
        print(f"\n💡 局域网测试：Flutter应用应使用地址：http://{ip_address}:8000")
        print(f"💡 安卓模拟器测试：Flutter应用应使用地址：http://10.0.2.2:8000")

        return True
    except Exception as e:
        print(f"❌ 获取网络信息失败：{e}")
        return False

def check_firewall():
    """
    检查Windows防火墙配置，提供端口放行解决方案
    （仅提供指引，不自动修改防火墙）
    """
    print_section("4. 检查Windows防火墙")

    print("🔍 检查防火墙规则...")
    print("\n如果防火墙拦截8000端口，请添加放行规则：")
    print("\n【方法1】PowerShell添加规则（需要管理员权限）：")
    print('  New-NetFirewallRule -DisplayName "Python API Server" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow')

    print("\n【方法2】图形界面配置：")
    print("  1. 打开Windows Defender防火墙 -> 高级设置")
    print("  2. 入站规则 -> 新建规则")
    print("  3. 规则类型：端口")
    print("  4. 协议：TCP，本地端口：8000")
    print("  5. 操作：允许连接")

    print("\n【方法3】临时关闭防火墙测试（生产环境不推荐）：")
    print("  控制面板 -> Windows Defender防火墙 -> 启用或关闭Windows Defender防火墙")

def test_api_with_image():
    """
    测试图片识别API核心功能，自动查找测试图片并发送请求
    :return: 识别成功返回True，失败返回False
    """
    print_section("5. 测试图片识别功能")

    # 获取项目根目录，递归查找测试图片（test开头的jpg/png）
    project_root = Path(__file__).parent.parent
    test_images = list(project_root.glob("**/test*.jpg")) + list(project_root.glob("**/test*.png"))

    if not test_images:
        print("⚠️  未找到测试图片")
        print("   建议：准备一张测试图片，命名为 test_image.jpg")
        return False

    # 使用找到的第一张测试图片
    test_image = test_images[0]
    print(f"📸 使用测试图片：{test_image}")

    try:
        # 以二进制方式打开图片，上传到识别接口
        with open(test_image, 'rb') as f:
            files = {'image': f}
            print("📤 发送识别请求...")
            response = requests.post(
                "http://127.0.0.1:8000/v1/detect_trash",
                files=files,
                timeout=60  # 模型推理较慢，设置60秒超时
            )

        if response.status_code == 200:
            data = response.json()
            print(f"✅ 识别成功！")
            print(f"   - 检测到目标数量：{data['detection_count']}")
            print(f"   - 推理耗时：{data['inference_time_ms']:.2f}ms")

            # 展示前3个识别结果
            if data['detections']:
                print(f"\n   识别结果：")
                for i, det in enumerate(data['detections'][:3]):
                    print(f"   [{i+1}] {det['specific_name']} ({det['general_category']}) - 置信度：{det['confidence']:.2f}")

            return True
        else:
            print(f"❌ 识别失败，状态码：{response.status_code}")
            print(f"   响应内容：{response.text[:200]}")
            return False

    except requests.exceptions.Timeout:
        print(f"❌ 识别请求超时（超过60秒）")
        print(f"   可能原因：")
        print(f"   1. CPU推理速度过慢")
        print(f"   2. 图片尺寸过大，处理耗时较长")
        print(f"   3. 服务器资源不足")
        return False
    except Exception as e:
        print(f"❌ 识别过程发生错误：{e}")
        return False

def provide_solutions():
    """
    汇总常见问题及解决方案，方便用户快速排查故障
    """
    print_section("常见问题解决方案")

    print("\n【问题1】端口未监听")
    print("  解决方法：确保API服务正在运行")
    print("  启动命令：cd api && python main.py")

    print("\n【问题2】模型加载超时")
    print("  解决方法：首次启动模型加载需要时间，请耐心等待")
    print("  提示：可在API服务控制台查看加载进度")

    print("\n【问题3】推理耗时过长")
    print("  原因：CPU环境下模型推理速度慢")
    print("  解决方案：")
    print("  - 使用NVIDIA GPU加速")
    print("  - 使用轻量模型（yolov8n替代yolov8s）")
    print("  - 在Flutter客户端增加请求超时时间")

    print("\n【问题4】安卓模拟器无法连接")
    print("  检查项：")
    print("  1. API服务监听 0.0.0.0（而非 127.0.0.1）")
    print("  2. Flutter配置使用 http://10.0.2.2:8000")
    print("  3. Windows防火墙放行8000端口")

    print("\n【问题5】防火墙拦截")
    print("  解决方法：添加防火墙规则放行8000端口（参考第4部分）")

def main():
    """
    主函数：按顺序执行所有诊断流程，输出最终结果
    """
    print("\n" + "🔧" * 30)
    print("   安卓模拟器连接诊断工具")
    print("🔧" * 30)

    # 存储所有诊断结果
    results = []

    # 1. 检查端口监听
    results.append(check_port_listening())

    # 2. 检查API健康状态
    results.append(check_localhost_api())

    # 3. 检查网络接口
    results.append(check_network_interfaces())

    # 4. 检查防火墙配置
    check_firewall()

    # 5. 仅当前面检查通过时，测试图片识别功能
    if results[0] and results[1]:
        results.append(test_api_with_image())

    # 输出问题解决方案
    provide_solutions()

    # 诊断结果汇总
    print_section("诊断结果汇总")
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✅ 所有检查项均通过（{passed}/{total}）")
        print(f"\n🎉 API服务运行正常！")
        print(f"\n下一步操作：")
        print(f"1. 确保安卓模拟器中Flutter应用配置为：http://10.0.2.2:8000")
        print(f"2. 若仍超时，检查Windows防火墙设置")
        print(f"3. 在Flutter应用中测试连接")
    else:
        print(f"⚠️  部分检查项失败（{passed}/{total}）")
        print(f"\n请根据上述解决方案修复问题后，重新运行本脚本")

    print("\n" + "="*60)

# 程序入口
if __name__ == "__main__":
    main()
