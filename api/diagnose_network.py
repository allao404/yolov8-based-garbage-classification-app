"""
Network Diagnostics Tool for Garbage Classification API

This script diagnoses network connectivity issues between the Android emulator
and the garbage classification API server. 
"""

import socket
import requests
import subprocess
import sys
from pathlib import Path

def print_section(title):
    """Print section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def check_port_listening(port=8000):
    """Check if port is listening"""
    print_section("1. Check Port Listening Status")

    try:
        # Check if port is in use
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()

        if result == 0:
            print(f"✅ Port {port} is listening")
            return True
        else:
            print(f"❌ Port {port} is not listening")
            print(f"   Please confirm the API server is started: python api/main.py")
            return False
    except Exception as e:
        print(f"❌ Error checking port: {e}")
        return False

def check_localhost_api():
    """Check if local API is accessible"""
    print_section("2. Check Local API Health Status")

    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API service is running normally")
            print(f"   - Model loaded: {data.get('model_loaded')}")
            print(f"   - GPU available: {data.get('gpu_available')}")
            print(f"   - Device in use: {data.get('device_in_use')}")
            return True
        else:
            print(f"❌ API returned abnormal status code: {response.status_code}")
            return False
    except requests.exceptions.Timeout:
        print(f"❌ API request timeout")
        print(f"   API server may be loading the model, please wait and retry")
        return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to API server")
        print(f"   Please confirm the API server is started")
        return False
    except Exception as e:
        print(f"❌ Error checking API: {e}")
        return False

def check_network_interfaces():
    """Check network interfaces and IP addresses"""
    print_section("3. Check Network Interfaces")

    try:
        # Get local IP address
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)

        print(f"✅ Hostname: {hostname}")
        print(f"✅ Local IP: {ip_address}")
        print(f"\n💡 For LAN testing, Flutter app should use: http://{ip_address}:8000")
        print(f"💡 For Android emulator testing, Flutter app should use: http://10.0.2.2:8000")

        return True
    except Exception as e:
        print(f"❌ Error getting network information: {e}")
        return False

def check_firewall():
    """Check firewall settings (Windows)"""
    print_section("4. Check Windows Firewall")

    print("🔍 Checking firewall rules...")
    print("\nIf firewall is blocking port 8000, add a rule:")
    print("\n【Method 1】Add rule via PowerShell (requires admin privileges):")
    print('  New-NetFirewallRule -DisplayName "Python API Server" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow')

    print("\n【Method 2】Via GUI:")
    print("  1. Open Windows Defender Firewall -> Advanced Settings")
    print("  2. Inbound Rules -> New Rule")
    print("  3. Rule Type: Port")
    print("  4. Protocol: TCP, Local Port: 8000")
    print("  5. Action: Allow the connection")

    print("\n【Method 3】Temporarily disable firewall for testing (not recommended for production):")
    print("  Control Panel -> Windows Defender Firewall -> Turn Windows Defender Firewall on or off")

def test_api_with_image():
    """Test API image detection functionality"""
    print_section("5. Test Image Detection")

    # Find test image
    project_root = Path(__file__).parent.parent
    test_images = list(project_root.glob("**/test*.jpg")) + list(project_root.glob("**/test*.png"))

    if not test_images:
        print("⚠️  No test image found")
        print("   Suggestion: Prepare a test image, name it test_image.jpg")
        return False

    test_image = test_images[0]
    print(f"📸 Using test image: {test_image}")

    try:
        with open(test_image, 'rb') as f:
            files = {'image': f}
            print("📤 Sending detection request...")
            response = requests.post(
                "http://127.0.0.1:8000/v1/detect_trash",
                files=files,
                timeout=60
            )

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Detection successful!")
            print(f"   - Detected {data['detection_count']} object(s)")
            print(f"   - Inference time: {data['inference_time_ms']:.2f}ms")

            if data['detections']:
                print(f"\n   Detection results:")
                for i, det in enumerate(data['detections'][:3]):  # Show only first 3
                    print(f"   [{i+1}] {det['specific_name']} ({det['general_category']}) - Confidence: {det['confidence']:.2f}")

            return True
        else:
            print(f"❌ Detection failed, status code: {response.status_code}")
            print(f"   Response content: {response.text[:200]}")
            return False

    except requests.exceptions.Timeout:
        print(f"❌ Detection request timeout (>60 seconds)")
        print(f"   Possible causes:")
        print(f"   1. Model inference too slow (CPU inference)")
        print(f"   2. Image too large, requires processing time")
        print(f"   3. Insufficient server resources")
        return False
    except Exception as e:
        print(f"❌ Error during detection: {e}")
        return False

def provide_solutions():
    """Provide solutions to common problems"""
    print_section("Common Problem Solutions")

    print("\n【Problem 1】Port Not Listening")
    print("  Solution: Ensure API server is running")
    print("  Command: cd api && python main.py")

    print("\n【Problem 2】Model Loading Timeout")
    print("  Solution: Model loading takes time on first startup, please be patient")
    print("  Note: You can see loading progress in API server logs")

    print("\n【Problem 3】Inference Time Too Long")
    print("  Cause: CPU inference is slow")
    print("  Solutions:")
    print("  - Use GPU acceleration (if NVIDIA GPU available)")
    print("  - Use smaller model (yolov8n instead of yolov8s)")
    print("  - Increase timeout in Flutter client")

    print("\n【Problem 4】Android Emulator Cannot Connect")
    print("  Check:")
    print("  1. API server listening on 0.0.0.0 (not 127.0.0.1)")
    print("  2. Flutter config using http://10.0.2.2:8000")
    print("  3. Windows firewall allows port 8000")

    print("\n【Problem 5】Firewall Blocking")
    print("  Solution: Add firewall rule to allow port 8000 (see Section 4 above)")

def main():
    """Main function"""
    print("\n" + "🔧" * 30)
    print("   Android Emulator Connection Diagnostic Tool")
    print("🔧" * 30)

    results = []

    # 1. Check port
    results.append(check_port_listening())

    # 2. Check API health status
    results.append(check_localhost_api())

    # 3. Check network interfaces
    results.append(check_network_interfaces())

    # 4. Check firewall
    check_firewall()

    # 5. Test image detection
    if results[0] and results[1]:  # If previous checks passed
        results.append(test_api_with_image())

    # Provide solutions
    provide_solutions()

    # Summary
    print_section("Diagnostic Summary")
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✅ All checks passed ({passed}/{total})")
        print(f"\n🎉 API server is running normally!")
        print(f"\nNext steps:")
        print(f"1. Ensure Flutter app in Android emulator is configured for: http://10.0.2.2:8000")
        print(f"2. If still timing out, check Windows firewall settings")
        print(f"3. Try testing connection in Flutter app")
    else:
        print(f"⚠️  Some checks failed ({passed}/{total})")
        print(f"\nPlease fix issues according to the solutions above and rerun this script")

    print("\n" + "="*60)

if __name__ == "__main__":
    main()
