# Garbage Classification Mobile App

A Flutter-based mobile application for AI-powered garbage classification using the YOLOv8 model for real-time object detection and classification.

## Features

### Core Features
- **AI-Powered Recognition**: Real-time garbage detection and classification using YOLOv8
- **Camera Capture**: Take photos with the device camera for instant recognition
- **Gallery Selection**: Choose images from the device photo library
- **Result Visualization**: Draw bounding boxes and labels on detected objects
- **Classification Guide**: Detailed garbage sorting instructions and explanations
- **Search Functionality**: Quickly search for waste types and disposal methods

### Supported Waste Types
- **BIODEGRADABLE** (Organic Waste) - Decomposable organic matter
- **CARDBOARD** (Cardboard) - Cardboard boxes and packaging
- **GLASS** (Glass) - Glass bottles, jars, and containers
- **METAL** (Metal) - Metal cans and containers
- **PAPER** (Paper) - Paper products and documents
- **PLASTIC** (Plastic) - Plastic bottles, bags, and containers

### Classification Categories
- **Recycle** (Recyclables) - Blue
- **Organic** (Organic Waste) - Green
- **Trash** (General Waste) - Gray
- **Hazardous** (Hazardous Waste) - Red

## Tech Stack

- **Frontend Framework**: Flutter 3.0+
- **Image Picker**: image_picker ^1.0.7
- **HTTP Client**: dio ^5.4.0
- **Permissions**: permission_handler ^11.0.0
- **Backend API**: FastAPI + YOLOv8
- **AI Model**: YOLOv8 (trained on Roboflow Garbage Classification dataset)

## Project Structure

```
mobile_app/
├── lib/
│   ├── models/              # Data models
│   │   ├── detection.dart   # Detection result model
│   │   └── garbage_guide.dart  # Classification guide model
│   ├── services/            # Service layer
│   │   ├── api_service.dart     # API service
│   │   ├── guide_service.dart   # Guide service
│   │   └── image_picker_service.dart  # Image picker service
│   ├── screens/             # Screens
│   │   ├── home_screen.dart     # Home screen
│   │   ├── detection_screen.dart  # Detection screen
│   │   └── guide_screen.dart    # Guide screen
│   ├── widgets/             # UI components
│   │   └── detection_painter.dart  # Bounding box painter
│   └── main.dart           # App entry point
├── android/                # Android configuration
├── ios/                    # iOS configuration
├── pubspec.yaml           # Dependencies
└── README.md              # Project documentation
```

## Core File Descriptions

### App Entry
- **lib/main.dart**: Application entry point, configures theme and routing

### Data Models
- **lib/models/detection.dart**:
  - `Detection`: Single detection result model
  - `DetectionResponse`: API response model

- **lib/models/garbage_guide.dart**:
  - `GarbageCategory`: Waste category model
  - `GarbageItem`: Specific waste item model

### Services
- **lib/services/api_service.dart**:
  - `GarbageDetectorService`: API communication service
  - Methods: `detectGarbage()`, `checkHealth()`, `getCategories()`

- **lib/services/guide_service.dart**:
  - `GuideService`: Garbage classification guide data service
  - Methods: `getAllCategories()`, `getAllItems()`, `searchItems()`

- **lib/services/image_picker_service.dart**:
  - `ImagePickerService`: Image selection service
  - Methods: `pickFromCamera()`, `pickFromGallery()`

### Screens
- **lib/screens/home_screen.dart**:
  - `HomeScreen`: Main screen with bottom navigation
  - `AboutScreen`: About page

- **lib/screens/detection_screen.dart**:
  - `DetectionScreen`: Main garbage detection interface
  - Features: Take photo, select image, display results, draw bounding boxes

- **lib/screens/guide_screen.dart**:
  - `GuideScreen`: Garbage classification guide page
  - Features: Display category info, search functionality

### Widgets
- **lib/widgets/detection_painter.dart**:
  - `DetectionPainter`: Custom painter for drawing bounding boxes
  - `DetectionOverlay`: Overlay component for detection results

## Installation & Setup

### Prerequisites
- Flutter SDK 3.0 or higher
- Android Studio or Xcode (depending on target platform)
- Backend API service deployed and running

### Setup Steps

1. **Navigate to Project**
```bash
cd /nas03/yixuh/garbage-classification/mobile_app
```

2. **Install Dependencies**
```bash
flutter pub get
```

3. **Configure API Endpoint**

Edit `lib/services/api_service.dart` and update the API URL:

```dart
static const String defaultApiUrl = "http://YOUR_SERVER_IP:8000";
```

#### Development Environment (LAN Testing)
- Ensure phone and API server are on the same local network
- Get server IP: `ifconfig` (Linux/Mac) or `ipconfig` (Windows)
- Example: `http://192.168.1.10:8000`

#### Production Environment
- Deploy API to a cloud server (Alibaba Cloud, Tencent Cloud, AWS, etc.)
- Use public domain or IP
- Example: `https://api.yourdomain.com`

4. **Run the App**

Android:
```bash
flutter run
```

iOS:
```bash
flutter run
```

Build APK (Android):
```bash
flutter build apk --release
```

Build IPA (iOS):
```bash
flutter build ios --release
```

## Permissions

### Android Permissions
The app requires:
- `CAMERA` - For taking photos
- `READ_EXTERNAL_STORAGE` - To read from gallery
- `WRITE_EXTERNAL_STORAGE` - To save images
- `INTERNET` - For network requests
- `ACCESS_NETWORK_STATE` - To check network status

### iOS Permissions
The app requires:
- `NSCameraUsageDescription` - Camera usage description
- `NSPhotoLibraryUsageDescription` - Photo library access description
- `NSPhotoLibraryAddUsageDescription` - Save to photo library description

## How to Use

### 1. Garbage Recognition

#### Camera Capture
1. Open the app, go to the "Detect" tab
2. Tap "Take Photo"
3. Grant camera permission
4. Capture a photo of the garbage
5. Wait for AI analysis
6. View detected waste types and disposal suggestions

#### Gallery Selection
1. Open the app, go to the "Detect" tab
2. Tap "Choose from Gallery"
3. Grant photo library permission
4. Select an image
5. Wait for AI analysis

### 2. Classification Guide

1. Switch to the "Guide" tab
2. Browse waste categories and instructions
3. Use the search bar to find specific items
4. Expand category cards for detailed information

### 3. API Settings

1. On the "Detect" screen, tap the settings icon in the top-right
2. Enter the API server address
3. Tap Save
4. The app will use the new endpoint

## API Endpoints

### Detection Endpoint
- **URL**: `POST /v1/detect_trash`
- **Content-Type**: `multipart/form-data`
- **Parameter**:
  - `image`: Image file

- **Response Example**:
```json
{
  "status": "success",
  "detection_count": 2,
  "detections": [
    {
      "bbox_xyxy": [450, 150, 520, 300],
      "confidence": 0.92,
      "specific_name": "PLASTIC",
      "general_category": "Recycle"
    },
    {
      "bbox_xyxy": [120, 220, 140, 235],
      "confidence": 0.78,
      "specific_name": "METAL",
      "general_category": "Recycle"
    }
  ],
  "inference_time_ms": 45.23
}
```

### Health Check Endpoint
- **URL**: `GET /health`
- **Response**: API server status

### Categories Endpoint
- **URL**: `GET /v1/categories`
- **Response**: All supported classification categories

## Troubleshooting

### 1. Cannot Connect to API Server

**Error**: "Unable to connect to server. Please check API address and network"

**Solutions**:
- Verify phone and server are on the same network
- Confirm API is running: `curl http://YOUR_IP:8000/health`
- Check firewall settings
- Verify API URL is correct
- Test connectivity: `ping YOUR_SERVER_IP`

### 2. Camera/Gallery Permission Denied

**Error**: "Camera permission required to take photos"

**Solutions**:
- Android: Settings → Apps → Garbage AI Assistant → Permissions → Enable Camera & Storage
- iOS: Settings → Privacy → Camera/Photos → Enable access

### 3. Inaccurate Detection Results

**Possible Causes**:
- Poor lighting
- Blurry or unclear object
- Object too small or far away

**Tips**:
- Take photos in well-lit environments
- Capture front-facing, full view of the object
- Keep appropriate distance (1–2 meters)
- Ensure object occupies most of the frame

### 4. App Crashes or Lags

**Solutions**:
- Restart the app
- Clear app cache
- Ensure sufficient storage space
- Update to the latest version

## Development Guide

### Change API Endpoint

Edit `lib/services/api_service.dart`:
```dart
static const String defaultApiUrl = "http://YOUR_NEW_IP:8000";
```

### Add New Waste Type

Edit `lib/services/guide_service.dart`, add to `getAllItems()`:
```dart
GarbageItem(
  name: 'NEW_TYPE',
  category: 'Recycle',
  description: 'Description',
  examples: ['Example 1', 'Example 2'],
),
```

### Customize Theme Color

Edit `lib/main.dart`:
```dart
theme: ThemeData(
  primarySwatch: Colors.green, // Change to desired color
  ...
),
```

## Performance Optimization

1. **Image Compression**: App automatically resizes to 1920x1080, 85% quality
2. **Network Timeout**: Set to 30 seconds, adjustable as needed
3. **Cache Management**: Regularly clear temporary files
4. **Memory Optimization**: Release unused image resources promptly

## Deployment

### Development (Local Network)
1. Start backend API service
2. Get server LAN IP
3. Configure API endpoint
4. Ensure devices are on the same network
5. Run the app

### Production (Public Network)
1. Deploy backend API to cloud server
2. Set up domain and SSL certificate (HTTPS)
3. Update API endpoint to public URL
4. Build release version
5. Upload to app stores or distribution platforms

### Temporary Testing with ngrok
```bash
# Run on API server
ngrok http 8000

# Use generated URL in app
# https://xxxx-xxx-xxx-xxx.ngrok.io
```

## FAQ

**Q: Which platforms are supported?**  
A: Android 5.0+ and iOS 11.0+

**Q: How long does detection take?**  
A: Typically 100–500ms, depending on network and server performance

**Q: Can it work offline?**  
A: Currently requires internet; offline model support planned for future versions

**Q: Does it support batch detection?**  
A: Current version supports single-image detection with multiple objects

**Q: What is the detection accuracy?**  
A: Average accuracy 85%+, up to 90%+ in good lighting and clear images


## Future Plans

- [ ] Offline model support
- [ ] Batch image processing
- [ ] Detection history
- [ ] Eco-points system
- [ ] Nearby recycling point map
- [ ] Scheduled pickup reminders
- [ ] Multi-language support
- [ ] Dark mode
- [ ] Share functionality

## Contributing

Issues and Pull Requests are welcome!

## License

MIT License