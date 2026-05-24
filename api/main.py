"""
FastAPI Service for Garbage Classification
Provides API endpoint for real-time trash detection and classification
垃圾分类FastAPI服务，实现垃圾实时检测分类接口
"""

# 系统基础库导入
import os
import json
import io
import time
from pathlib import Path
from typing import List, Dict, Optional
import logging

# 图像处理与深度学习库导入
import numpy as np
import cv2
from PIL import Image
# FastAPI框架相关组件
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
# 数据模型校验
from pydantic import BaseModel, Field
# YOLO检测模型
from ultralytics import YOLO
# 深度学习硬件加速
import torch


# ====================== 日志配置模块 ======================
# 自定义日志输出格式，记录时间、模块、级别、代码行号与日志信息
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


# ====================== 接口响应数据模型 ======================
class Detection(BaseModel):
    """单目标垃圾检测结果数据模型"""
    bbox_xyxy: List[float] = Field(
        ...,
        description="检测框坐标 [左上角x,左上角y,右下角x,右下角y]"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="目标检测置信度分数"
    )
    specific_name: str = Field(
        ...,
        description="垃圾细分类别名称（一级标签）"
    )
    general_category: str = Field(
        ...,
        description="垃圾大类名称（二级标签）：可回收/干垃圾/有害垃圾/厨余垃圾"
    )


class DetectionResponse(BaseModel):
    """垃圾检测接口统一返回数据模型"""
    status: str = Field(
        default="success",
        description="接口请求状态"
    )
    detection_count: int = Field(
        ...,
        ge=0,
        description="图片内检测到的垃圾数量"
    )
    detections: List[Detection] = Field(
        default_factory=list,
        description="所有垃圾目标检测结果列表"
    )
    inference_time_ms: float = Field(
        ...,
        description="模型推理耗时，单位毫秒"
    )


class ErrorResponse(BaseModel):
    """接口异常错误返回数据模型"""
    status: str = "error"
    message: str
    detail: Optional[str] = None


# ====================== FastAPI服务初始化 ======================
# 创建FastAPI应用实例，配置服务基础信息
app = FastAPI(
    title="Garbage Classification API",
    description="基于YOLOv8的实时垃圾检测分类接口服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 跨域中间件配置，允许移动端、前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境放行所有跨域来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ====================== 请求日志中间件 ======================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """全局请求拦截中间件，记录请求信息与处理耗时"""
    start_time = time.time()
    client_host = request.client.host if request.client else "unknown"

    logger.info(f"➡️  收到请求: {request.method} {request.url.path} 客户端地址: {client_host}")

    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        logger.info(f"✅ 请求处理完成 耗时{process_time:.2f}ms - 响应状态码: {response.status_code}")
        return response
    except Exception as e:
        process_time = (time.time() - start_time) * 1000
        logger.error(f"❌ 请求处理异常 耗时{process_time:.2f}ms - 错误信息: {str(e)}")
        raise


# ====================== 全局变量定义 ======================
model = None                # YOLO检测模型全局实例
category_mapping = None     # 垃圾类别映射字典
device = 'cuda'             # 模型运行硬件设备


# ====================== 模型加载函数 ======================
def load_model(model_path: str = None):
    """
    加载YOLOv8垃圾检测模型
    :param model_path: 模型权重文件路径，为空则使用默认路径
    :return: 加载完成的模型实例
    """
    global model, device

    # 未传入路径则自动拼接项目默认模型路径
    if model_path is None:
        project_root = Path(__file__).parent.parent
        model_path = project_root / "models" / "garbage_yolov8s" / "weights" / "best.pt"

    model_path = Path(model_path)

    # 校验模型文件是否存在
    if not model_path.exists():
        logger.error(f"模型文件不存在: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"开始加载模型: {model_path}")

    try:
        # 自动判断硬件，优先使用GPU加速
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"检测到可用GPU: {gpu_name}")
        else:
            device = 'cpu'
            logger.warning("无可用GPU，使用CPU进行推理，速度较慢")

        # 初始化模型并绑定运行设备
        model = YOLO(str(model_path))
        model.to(device)

        logger.info(f"模型加载成功，运行设备: {device}")
        return model

    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise


# ====================== 类别映射加载函数 ======================
def load_category_mapping(mapping_path: str = None):
    """
    加载垃圾细分类别与大类的映射配置文件
    :param mapping_path: 映射json文件路径，为空使用默认路径
    :return: 类别映射字典
    """
    global category_mapping

    # 未传入路径则使用默认配置文件路径
    if mapping_path is None:
        project_root = Path(__file__).parent.parent
        mapping_path = project_root / "configs" / "category_mapping.json"

    mapping_path = Path(mapping_path)

    # 校验配置文件存在性
    if not mapping_path.exists():
        logger.error(f"类别映射文件不存在: {mapping_path}")
        raise FileNotFoundError(f"Category mapping file not found: {mapping_path}")

    logger.info(f"加载类别映射配置: {mapping_path}")

    try:
        # 读取json映射数据
        with open(mapping_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            category_mapping = data['mapping']

        logger.info(f"类别映射加载完成，共{len(category_mapping)}个细分垃圾类别")
        return category_mapping

    except Exception as e:
        logger.error(f"类别映射文件读取失败: {e}")
        raise


# ====================== 服务启动初始化事件 ======================
@app.on_event("startup")
async def startup_event():
    """服务启动钩子，初始化模型与类别映射数据"""
    logger.info("="*60)
    logger.info("垃圾分类检测API服务启动")
    logger.info("="*60)

    try:
        # 加载检测模型
        load_model()
        # 加载类别映射关系
        load_category_mapping()

        logger.info("API服务初始化全部完成！")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"服务初始化失败: {e}")
        logger.error("请检查模型权重文件与类别配置文件是否齐全")
        raise


# ====================== 健康检查接口 ======================
@app.get("/", tags=["Health"])
async def root():
    """根路径接口，简易服务状态检测"""
    return {
        "service": "Garbage Classification API",
        "status": "running",
        "version": "1.0.0",
        "model": "YOLOv8m",
        "endpoints": {
            "detection": "/v1/detect_trash",
            "docs": "/docs",
            "health": "/health"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """详细健康自检接口，检测模型、硬件、配置加载状态"""
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "N/A"

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "category_mapping_loaded": category_mapping is not None,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "device_in_use": device if device else "N/A"
    }


# ====================== 核心垃圾检测接口 ======================
@app.post(
    "/v1/detect_trash",
    response_model=DetectionResponse,
    responses={
        200: {"description": "垃圾检测成功"},
        400: {"description": "请求图片参数非法"},
        500: {"description": "服务内部推理异常"}
    },
    tags=["Detection"]
)
async def detect_trash(
    image: UploadFile = File(..., description="待检测的图片文件")
):
    """
    上传图片完成垃圾检测与分类
    返回：垃圾数量、检测框坐标、置信度、细分类别、垃圾大类、推理耗时
    """
    # 校验上传文件格式，仅允许图片类型
    if not image.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"文件类型非法: {image.content_type}，请上传图片格式文件"
        )

    try:
        # 读取上传图片二进制数据
        logger.info(f"📥 接收上传图片: {image.filename} 文件格式: {image.content_type}")
        contents = await image.read()
        file_size_mb = len(contents) / (1024 * 1024)
        logger.info(f"📦 图片文件大小: {file_size_mb:.2f} MB")

        image_bytes = io.BytesIO(contents)

        # 二进制流转PIL图像对象
        logger.info("🖼️  解析图片图像数据")
        pil_image = Image.open(image_bytes)

        # 图像转为numpy数组，适配模型输入格式
        logger.info("🔄 图像格式转换为模型输入数组")
        img_array = np.array(pil_image)

        # 剔除透明Alpha通道，保留RGB三通道图像
        if img_array.shape[-1] == 4:
            logger.info("🎨 去除图片透明通道")
            img_array = img_array[..., :3]

        logger.info(f"✅ 图像预处理完成 | 图像尺寸: {img_array.shape} | 运行设备: {device}")

        # 执行模型推理检测
        start_time = time.time()
        logger.info(f"🤖 开始{device}端模型推理")
        # 设定置信度阈值、IOU非极大抑制阈值
        results = model(img_array, conf=0.25, iou=0.45, device=device)

        # 计算推理耗时
        inference_time = (time.time() - start_time) * 1000
        logger.info(f"⚡ 模型推理结束，耗时{inference_time:.2f}ms")

        # 解析推理结果，封装检测数据
        detections = []
        for result in results:
            boxes = result.boxes
            # 遍历所有检测到的垃圾目标
            for box in boxes:
                # 获取检测框坐标
                bbox = box.xyxy[0].cpu().numpy().tolist()
                # 获取检测置信度
                confidence = float(box.conf[0].cpu().numpy())
                # 获取类别ID与细分名称
                class_id = int(box.cls[0].cpu().numpy())
                specific_name = model.names[class_id]
                # 匹配对应垃圾大类
                general_category = category_mapping.get(specific_name, "Unknown")

                # 组装单条检测结果
                detection = Detection(
                    bbox_xyxy=bbox,
                    confidence=confidence,
                    specific_name=specific_name,
                    general_category=general_category
                )
                detections.append(detection)

        logger.info(
            f"图片检测完成：共发现{len(detections)}个垃圾目标 | "
            f"推理耗时: {inference_time:.2f}ms"
        )

        # 构造接口统一响应体
        response = DetectionResponse(
            status="success",
            detection_count=len(detections),
            detections=detections,
            inference_time_ms=round(inference_time, 2)
        )
        return response

    except Exception as e:
        logger.error(f"图片检测处理异常: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"图像检测处理失败: {str(e)}"
        )


# ====================== 类别信息查询接口 ======================
@app.get("/v1/categories", tags=["Info"])
async def get_categories():
    """查询服务支持的所有垃圾类别及分类统计信息"""
    if category_mapping is None:
        raise HTTPException(status_code=500, detail="类别映射数据未加载")

    # 统计各大类垃圾数量
    category_counts = {}
    for specific, general in category_mapping.items():
        category_counts[general] = category_counts.get(general, 0) + 1

    return {
        "total_classes": len(category_mapping),
        "specific_categories": list(category_mapping.keys()),
        "general_categories": {
            "Recycle": category_counts.get("Recycle", 0),
            "Trash": category_counts.get("Trash", 0),
            "Hazardous": category_counts.get("Hazardous", 0),
            "Organic": category_counts.get("Organic", 0)
        },
        "mapping": category_mapping
    }


# ====================== 服务入口启动程序 ======================
if __name__ == "__main__":
    import uvicorn

    # 启动uvicorn服务器，监听全网8000端口
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
