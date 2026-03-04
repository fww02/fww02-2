"""
Academic-Grade Scene Graph Visualization for 3D-MEM
Dual-Mode Visualizer: Textured Mode + Topology Mode

Features:
- Textured Mode: Habitat 渲染纹理 + 实时位置标注
- Topology Mode: 层级拓扑图（带决策标注）
- 强制 2D 投影（XZ 平面），修复 shapes (2,) (3,) 错误
- 精确坐标对齐（world_to_pixel）
- 无头环境适配（Matplotlib Agg 后端）
- 等比例坐标（axis('equal')）
- 支持外部高质量 top-down map 加载
"""

import os
import logging
from typing import Any, Dict, Optional, List, Tuple, Literal, Union
from collections import defaultdict
from enum import Enum
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 强制无头渲染，必须在 import pyplot 之前
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

logger = logging.getLogger(__name__)


class VisualizationMode(Enum):
    """可视化模式枚举"""
    TEXTURED = "textured"      # 真实纹理俯视图（Habitat 渲染）
    TOPOLOGY = "topology"      # 层级拓扑图（带决策标注）


def load_external_topdown_map(
    image_path: Union[str, Path],
    target_size: Optional[Tuple[int, int]] = None,
) -> Optional[np.ndarray]:
    """
    加载外部高质量 top-down map 图像
    
    支持格式: PNG, JPG, JPEG, BMP, TIFF
    
    Args:
        image_path: 外部图像文件路径
        target_size: 可选的目标尺寸 (width, height)，如果提供则会调整大小
        
    Returns:
        RGB 图像数组 (H, W, 3) uint8，如果加载失败返回 None
    """
    try:
        import cv2
        
        path = Path(image_path)
        if not path.exists():
            logger.warning(f"External top-down map not found: {image_path}")
            return None
        
        # 读取图像 (OpenCV 使用 BGR)
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning(f"Failed to load image: {image_path}")
            return None
        
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 调整大小（如果需要）
        if target_size is not None:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        
        logger.info(f"Loaded external top-down map: {image_path}, shape={img.shape}")
        return img
        
    except Exception as e:
        logger.warning(f"Failed to load external top-down map: {e}")
        return None


class SceneGraphVisualizer:
    """
    学术级场景图可视化器
    
    双模式独立渲染：
    - Textured Mode: 外部注入 Habitat top_down_map，实时标注位置
    - Topology Mode: 层级拓扑图，带决策冲突标注
    
    支持外部高质量底图：
    - 可通过 load_external_topdown_map() 函数加载外部图像
    - 支持自定义底图边界用于坐标转换
    """

    # ==================== 论文级配色方案 ====================
    # 基于学术论文常用配色
    COLORS = {
        # 节点颜色
        'snapshot': '#3498DB',          # 浅蓝色 - 图像节点 (Snapshot)
        'snapshot_edge': '#2980B9',     # 深蓝色 - 图像节点边框
        'object': '#E74C3C',            # 红色 - 物体节点
        'object_edge': '#C0392B',       # 深红色 - 物体节点边框
        'agent': '#2ECC71',             # 绿色 - 当前位置
        'agent_edge': '#27AE60',        # 深绿色 - 当前位置边框
        'target': '#9B59B6',            # 紫色 - 目标节点
        'target_halo': '#9B59B640',     # 半透明紫色光晕
        
        # 连线颜色
        'trajectory': '#3498DB80',      # 半透明蓝色 - 轨迹线
        'association': '#E74C3C40',     # 半透明红色 - 物体关联线
        
        # 背景/辅助颜色
        'background': '#FAFAFA',        # 浅灰色背景
        'grid': '#E0E0E0',              # 网格线
        'text': '#2C3E50',              # 深灰色文字
        'label_bg': '#00000080',        # 半透明黑色标签背景
    }

    def __init__(
        self, 
        voxel_size: float = 0.1,
        map_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        style: str = "academic",  # "academic" or "default"
    ):
        """
        Args:
            voxel_size: 体素大小（米）
            map_bounds: 地图边界 (min_bound, max_bound)，每个为 (3,) 数组 [x_min, y_min, z_min]
            style: 渲染风格 ("academic" 用于论文, "default" 用于调试)
        """
        self.voxel_size = voxel_size
        self.map_bounds = map_bounds
        self.style = style
        self.logger = logging.getLogger(__name__)
        
        # 冲突决策颜色映射（仅 Topology 模式使用）
        self.decision_colors = {
            "MERGE": "#95A5A6",   # 灰色
            "SPLIT": "#E74C3C",   # 红色
            "REPLACE": "#F39C12", # 橙色
            "KEEP": "#F1C40F",    # 黄色
        }
        
        # 设置 matplotlib 全局样式
        if style == "academic":
            plt.rcParams.update({
                'font.family': 'sans-serif',
                'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
                'font.size': 10,
                'axes.labelsize': 12,
                'axes.titlesize': 14,
                'legend.fontsize': 9,
                'figure.titlesize': 16,
            })

    def set_map_bounds(
        self, 
        min_bound: np.ndarray, 
        max_bound: np.ndarray
    ) -> None:
        """
        设置地图边界（用于 world_to_pixel 坐标转换）
        
        Args:
            min_bound: 最小边界 (3,) [x_min, y_min, z_min]
            max_bound: 最大边界 (3,) [x_max, y_max, z_max]
        """
        self.map_bounds = (np.asarray(min_bound), np.asarray(max_bound))

    def world_to_pixel(
        self,
        world_pos: np.ndarray,
        map_shape: Tuple[int, int],
    ) -> Tuple[int, int]:
        """
        将 3D 世界坐标转换为 2D 纹理图像素坐标
        
        数学公式：
            pixel_x = (world_x - x_min) / (x_max - x_min) * map_width
            pixel_y = (world_z - z_min) / (z_max - z_min) * map_height
        
        Args:
            world_pos: 3D 世界坐标 (3,) [x, y, z] 或 2D (2,) [x, z]
            map_shape: 纹理图尺寸 (height, width)
            
        Returns:
            (pixel_x, pixel_y) 像素坐标
        """
        if self.map_bounds is None:
            # 无边界时使用 voxel_size 进行简单转换
            if len(world_pos) == 3:
                px = int(world_pos[0] / self.voxel_size)
                py = int(world_pos[2] / self.voxel_size)
            else:
                px = int(world_pos[0] / self.voxel_size)
                py = int(world_pos[1] / self.voxel_size)
            return (px, py)
        
        min_bound, max_bound = self.map_bounds
        map_height, map_width = map_shape
        
        # 提取 X 和 Z 坐标（Habitat 使用 Y-up 坐标系）
        if len(world_pos) == 3:
            world_x, world_z = world_pos[0], world_pos[2]
        else:
            world_x, world_z = world_pos[0], world_pos[1]
        
        # 归一化到 [0, 1]
        x_range = max_bound[0] - min_bound[0]
        z_range = max_bound[2] - min_bound[2]
        
        if x_range < 1e-6:
            x_range = 1.0
        if z_range < 1e-6:
            z_range = 1.0
        
        norm_x = (world_x - min_bound[0]) / x_range
        norm_z = (world_z - min_bound[2]) / z_range
        
        # 转换为像素坐标
        pixel_x = int(np.clip(norm_x * map_width, 0, map_width - 1))
        pixel_y = int(np.clip(norm_z * map_height, 0, map_height - 1))
        
        return (pixel_x, pixel_y)

    def pixel_to_world(
        self,
        pixel_pos: Tuple[int, int],
        map_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        将 2D 像素坐标转换为 3D 世界坐标（逆变换）
        
        Args:
            pixel_pos: (pixel_x, pixel_y)
            map_shape: (height, width)
            
        Returns:
            (3,) [world_x, 0.0, world_z]
        """
        if self.map_bounds is None:
            world_x = pixel_pos[0] * self.voxel_size
            world_z = pixel_pos[1] * self.voxel_size
            return np.array([world_x, 0.0, world_z])
        
        min_bound, max_bound = self.map_bounds
        map_height, map_width = map_shape
        
        norm_x = pixel_pos[0] / map_width
        norm_z = pixel_pos[1] / map_height
        
        world_x = norm_x * (max_bound[0] - min_bound[0]) + min_bound[0]
        world_z = norm_z * (max_bound[2] - min_bound[2]) + min_bound[2]
        
        return np.array([world_x, 0.0, world_z])

    def world_to_pixel_mpp(
        self,
        world_pos: np.ndarray,
        meters_per_pixel: float,
    ) -> Tuple[int, int]:
        """
        使用 meters_per_pixel 进行精确的世界到像素坐标转换
        
        这个方法专门为 get_structured_topdown_map 生成的底图设计，
        保证与底图生成时使用的坐标系统完全一致。
        
        数学公式：
            pixel_x = (world_x - min_bound[0]) / meters_per_pixel
            pixel_y = (world_z - min_bound[2]) / meters_per_pixel
        
        Args:
            world_pos: 3D 世界坐标 (3,) [x, y, z] 或 2D (2,) [x, z]
            meters_per_pixel: 分辨率（米/像素）
            
        Returns:
            (pixel_x, pixel_y) 像素坐标
        """
        if self.map_bounds is None:
            raise ValueError("map_bounds must be set before calling world_to_pixel_mpp")
        
        min_bound, max_bound = self.map_bounds
        
        # 提取 X 和 Z 坐标（Habitat 使用 Y-up 坐标系）
        if len(world_pos) == 3:
            world_x, world_z = world_pos[0], world_pos[2]
        else:
            world_x, world_z = world_pos[0], world_pos[1]
        
        # 直接使用 meters_per_pixel 转换（与底图生成完全一致）
        pixel_x = int((world_x - min_bound[0]) / meters_per_pixel)
        pixel_y = int((world_z - min_bound[2]) / meters_per_pixel)
        
        return (pixel_x, pixel_y)

    # ==================== Textured Mode ====================

    def visualize_textured(
        self,
        top_down_map: np.ndarray,
        output_path: str,
        agent_position: Optional[np.ndarray] = None,
        agent_heading: Optional[float] = None,
        object_positions: Optional[Dict[int, np.ndarray]] = None,
        object_classes: Optional[Dict[int, str]] = None,
        trajectory: Optional[np.ndarray] = None,
        snapshot_positions: Optional[List[np.ndarray]] = None,
        snapshot_connections: Optional[List[Tuple[int, int]]] = None,
        object_to_snapshot: Optional[Dict[int, int]] = None,
        target_object_id: Optional[int] = None,
        title: str = "Explicit Memory Graph",
        show_object_labels: bool = False,
        show_legend: bool = True,
        figsize: Tuple[float, float] = (20, 16),
        dpi: int = 300,
    ):
        """
        学术论文级俯视可视化 (Academic Top-Down Visualization)
        
        生成类似 Habitat/ScanNet 风格的显式记忆图（Explicit Memory Graph）俯视示意图。
        
        视觉规范：
        1️⃣ 图像节点（Image/Snapshot Nodes）: 蓝色实心圆圈，表示机器人 Snapshot 中心点
        2️⃣ 对象节点（Object Nodes）: 粉色等边三角形（尖角朝上），表示语义物体
        3️⃣ 目标节点（Goal Node）: 红色大圆圈 + 半透明红色光晕
        4️⃣ 当前位置（Current Position）: 鲜绿色实心圆
        5️⃣ 轨迹线（Trajectory）: 粗壮的浅蓝色半透明色带
        6️⃣ 关联线（Image–Object Relations）: 粉色半透明细线
        
        Args:
            top_down_map: 外部提供的高质量 RGB 俯视底图 (H, W, 3)
            output_path: 输出路径（.png）
            agent_position: 机器人当前 3D 位置 (3,) [x, y, z]
            agent_heading: 机器人朝向角度（弧度）
            object_positions: 物体位置 {obj_id: (3,) position}
            object_classes: 物体类别 {obj_id: class_name}
            trajectory: 机器人轨迹 (N, 3)
            snapshot_positions: Snapshot 位置列表 [(3,), ...]
            snapshot_connections: Snapshot 之间的连接 [(i, j), ...]
            object_to_snapshot: 物体到 Snapshot 的关联 {obj_id: snapshot_idx}
            target_object_id: 目标物体 ID（用于高亮显示）
            title: 图像标题
            show_object_labels: 是否显示物体类别标签（默认 False，保持简洁）
            show_legend: 是否显示图例（默认 True）
            figsize: 图像尺寸 (width, height) in inches
            dpi: 输出 DPI（默认 300，论文级质量）
        """
        if top_down_map is None or top_down_map.size == 0:
            self.logger.warning("No top_down_map provided for textured mode")
            return

        # ==================== 学术论文配色方案 ====================
        # 严格按照用户规范
        COLOR_IMAGE_NODE = '#4A90D9'        # 蓝色 - 图像/Snapshot 节点
        COLOR_IMAGE_NODE_EDGE = '#2C5282'   # 深蓝色边框
        COLOR_OBJECT_NODE = '#FF69B4'       # 粉色 - 对象节点
        COLOR_OBJECT_NODE_EDGE = '#DB7093'  # 深粉色边框
        COLOR_TRAJECTORY = '#87CEEB'        # 浅蓝色 - 轨迹线
        COLOR_ASSOCIATION = '#FFB6C1'       # 浅粉色 - 关联线
        COLOR_AGENT = '#32CD32'             # 鲜绿色 - 当前位置
        COLOR_AGENT_EDGE = '#228B22'        # 深绿色边框
        COLOR_TARGET = '#DC143C'            # 深红色 - 目标节点
        COLOR_TARGET_HALO = '#FF6B6B'       # 红色光晕

        # ==================== 图像设置 ====================
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        fig.patch.set_facecolor('white')
        
        # 绘制底图
        ax.imshow(top_down_map, origin="upper", interpolation='bilinear')
        map_shape = top_down_map.shape[:2]  # (H, W)
        
        # 动态计算节点大小（基于地图尺寸，确保视觉协调）
        map_diag = np.sqrt(map_shape[0]**2 + map_shape[1]**2)
        base_radius = max(10, min(20, map_diag / 80))  # 中等大小节点
        trajectory_width = base_radius * 1.2  # 粗壮轨迹线
        
        # ==================== Layer 1: 轨迹线（底层）====================
        # 在相邻蓝色图像节点之间绘制粗壮的浅蓝色半透明色带
        if snapshot_positions is not None and len(snapshot_positions) > 1:
            snapshot_pixels = []
            for pos in snapshot_positions:
                px, py = self.world_to_pixel(pos, map_shape)
                snapshot_pixels.append((px, py))
            snapshot_pixels = np.array(snapshot_pixels)
            
            if snapshot_connections is not None:
                # 使用指定的连接关系
                for (i, j) in snapshot_connections:
                    if i < len(snapshot_pixels) and j < len(snapshot_pixels):
                        ax.plot(
                            [snapshot_pixels[i, 0], snapshot_pixels[j, 0]],
                            [snapshot_pixels[i, 1], snapshot_pixels[j, 1]],
                            color=COLOR_TRAJECTORY,
                            linewidth=trajectory_width,
                            alpha=0.65,
                            solid_capstyle='round',
                            solid_joinstyle='round',
                            zorder=2
                        )
            else:
                # 默认顺序连接
                ax.plot(
                    snapshot_pixels[:, 0], snapshot_pixels[:, 1],
                    color=COLOR_TRAJECTORY,
                    linewidth=trajectory_width,
                    alpha=0.65,
                    solid_capstyle='round',
                    solid_joinstyle='round',
                    zorder=2
                )

        # ==================== Layer 2: 关联线（Image–Object Relations）====================
        # 从蓝色图像节点出发，使用粉色细线连接到对应的粉色三角形对象节点
        if object_positions and object_to_snapshot and snapshot_positions:
            for obj_id, snapshot_idx in object_to_snapshot.items():
                # 跳过目标物体
                if target_object_id is not None and obj_id == target_object_id:
                    continue
                    
                if obj_id in object_positions and snapshot_idx < len(snapshot_positions):
                    obj_pos = object_positions[obj_id]
                    snap_pos = snapshot_positions[snapshot_idx]
                    
                    obj_px, obj_py = self.world_to_pixel(obj_pos, map_shape)
                    snap_px, snap_py = self.world_to_pixel(snap_pos, map_shape)
                    
                    ax.plot(
                        [snap_px, obj_px], [snap_py, obj_py],
                        color=COLOR_ASSOCIATION,
                        linewidth=1.8,
                        alpha=0.55,
                        solid_capstyle='round',
                        zorder=3
                    )

        # ==================== Layer 3: 图像节点（蓝色实心圆圈）====================
        # 表示机器人在运动过程中采集的 Snapshot 关键帧中心点
        if snapshot_positions is not None:
            for i, pos in enumerate(snapshot_positions):
                px, py = self.world_to_pixel(pos, map_shape)
                
                # 蓝色实心圆圈，边缘圆润
                circle = plt.Circle(
                    (px, py), 
                    radius=base_radius,
                    facecolor=COLOR_IMAGE_NODE,
                    edgecolor=COLOR_IMAGE_NODE_EDGE,
                    linewidth=2.5,
                    alpha=0.92,
                    zorder=10
                )
                ax.add_patch(circle)

        # ==================== Layer 4: 对象节点（粉色等边三角形）====================
        # 尖角朝上，表示从视觉中识别出的具体语义物体
        background_classes = {'wall', 'floor', 'ceiling', 'paneling', 'banner', 'misc', 'unknown'}
        
        if object_positions:
            for obj_id, pos in object_positions.items():
                # 跳过目标物体（单独绘制）
                if target_object_id is not None and obj_id == target_object_id:
                    continue
                
                # 过滤背景类
                class_name = object_classes.get(obj_id, '') if object_classes else ''
                if class_name.lower() in background_classes:
                    continue
                
                px, py = self.world_to_pixel(pos, map_shape)
                
                # 等边三角形（尖角朝上）
                triangle_size = base_radius * 0.85
                # 顶点朝上的等边三角形顶点坐标
                triangle = plt.Polygon(
                    [
                        (px, py - triangle_size * 1.15),  # 顶点（朝上）
                        (px - triangle_size, py + triangle_size * 0.58),  # 左下
                        (px + triangle_size, py + triangle_size * 0.58),  # 右下
                    ],
                    facecolor=COLOR_OBJECT_NODE,
                    edgecolor=COLOR_OBJECT_NODE_EDGE,
                    linewidth=2.0,
                    alpha=0.88,
                    zorder=15
                )
                ax.add_patch(triangle)
                
                # 可选：类别标签
                if show_object_labels and class_name and class_name.lower() not in background_classes:
                    ax.text(
                        px, py + triangle_size * 1.5,
                        class_name,
                        fontsize=max(6, base_radius * 0.45),
                        color='#2C3E50',
                        fontweight='semibold',
                        ha='center',
                        va='top',
                        bbox=dict(
                            boxstyle='round,pad=0.15',
                            facecolor='white',
                            alpha=0.75,
                            edgecolor='#BDC3C7',
                            linewidth=0.8
                        ),
                        zorder=16
                    )

        # ==================== Layer 5: 目标节点（红色大圆圈 + 光晕）====================
        if target_object_id is not None and object_positions and target_object_id in object_positions:
            pos = object_positions[target_object_id]
            px, py = self.world_to_pixel(pos, map_shape)
            
            # 半透明红色光晕（多层渐变，强调效果）
            for r_mult, alpha in [(3.5, 0.12), (2.8, 0.18), (2.2, 0.25), (1.7, 0.32)]:
                halo = plt.Circle(
                    (px, py),
                    radius=base_radius * r_mult,
                    facecolor=COLOR_TARGET_HALO,
                    edgecolor='none',
                    alpha=alpha,
                    zorder=17
                )
                ax.add_patch(halo)
            
            # 红色大圆圈（明显大于普通图像节点）
            target_circle = plt.Circle(
                (px, py),
                radius=base_radius * 1.4,
                facecolor=COLOR_TARGET,
                edgecolor='white',
                linewidth=3.5,
                alpha=0.95,
                zorder=19
            )
            ax.add_patch(target_circle)
            
            # 目标标签
            if object_classes and target_object_id in object_classes:
                target_name = object_classes[target_object_id]
                ax.text(
                    px, py - base_radius * 3.0,
                    f"Goal: {target_name}",
                    fontsize=max(8, base_radius * 0.55),
                    color='white',
                    fontweight='bold',
                    ha='center',
                    bbox=dict(
                        boxstyle='round,pad=0.35',
                        facecolor=COLOR_TARGET,
                        alpha=0.92,
                        edgecolor='white',
                        linewidth=2
                    ),
                    zorder=20
                )

        # ==================== Layer 6: 当前机器人位置（鲜绿色实心圆）====================
        if agent_position is not None:
            px, py = self.world_to_pixel(agent_position, map_shape)
            
            # 鲜绿色实心圆（与蓝色图像节点明显区分）
            agent_circle = plt.Circle(
                (px, py),
                radius=base_radius * 1.25,
                facecolor=COLOR_AGENT,
                edgecolor=COLOR_AGENT_EDGE,
                linewidth=3.0,
                alpha=0.95,
                zorder=25
            )
            ax.add_patch(agent_circle)
            
            # 朝向箭头（可选）
            if agent_heading is not None:
                arrow_len = base_radius * 2.8
                # Habitat 坐标系：heading 为相对于 Z 轴的角度
                dx = arrow_len * np.sin(agent_heading)
                dy = -arrow_len * np.cos(agent_heading)  # Y 轴翻转
                ax.arrow(
                    px, py, dx, dy,
                    head_width=base_radius * 0.7,
                    head_length=base_radius * 0.45,
                    fc='#FFD700',  # 金黄色箭头
                    ec='#2C3E50',   # 深灰边框
                    linewidth=1.8,
                    zorder=26
                )

        # ==================== 图像属性设置 ====================
        ax.set_xlim(0, map_shape[1])
        ax.set_ylim(map_shape[0], 0)  # Y 轴翻转（图像坐标系）
        ax.set_aspect('equal')  # 强制等比例
        ax.axis('off')  # 隐藏坐标轴
        
        # ==================== 图例 ====================
        if show_legend:
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', 
                           markerfacecolor=COLOR_IMAGE_NODE, markersize=14, 
                           label='Image Node', markeredgecolor=COLOR_IMAGE_NODE_EDGE, 
                           markeredgewidth=2, linestyle='None'),
                plt.Line2D([0], [0], marker='^', color='w', 
                           markerfacecolor=COLOR_OBJECT_NODE, markersize=14, 
                           label='Object Node', markeredgecolor=COLOR_OBJECT_NODE_EDGE, 
                           markeredgewidth=2, linestyle='None'),
                plt.Line2D([0], [0], marker='o', color='w', 
                           markerfacecolor=COLOR_AGENT, markersize=14, 
                           label='Current Position', markeredgecolor=COLOR_AGENT_EDGE, 
                           markeredgewidth=2, linestyle='None'),
                plt.Line2D([0], [0], color=COLOR_TRAJECTORY, 
                           linewidth=8, alpha=0.65, label='Trajectory'),
            ]
            
            if target_object_id is not None:
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor=COLOR_TARGET, markersize=16, 
                               label='Goal', markeredgecolor='white', 
                               markeredgewidth=2.5, linestyle='None')
                )
            
            legend = ax.legend(
                handles=legend_elements,
                loc='upper right',
                fontsize=max(9, base_radius * 0.7),
                framealpha=0.92,
                fancybox=True,
                shadow=False,
                frameon=True,
                edgecolor='#CCCCCC',
                borderpad=0.8,
                labelspacing=0.8,
                handletextpad=0.6,
            )
            legend.get_frame().set_linewidth(1.2)

        # ==================== 保存高清图像 ====================
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.tight_layout(pad=0.3)
        plt.savefig(
            output_path, 
            dpi=dpi, 
            bbox_inches='tight', 
            facecolor='white', 
            edgecolor='none',
            pad_inches=0.1
        )
        plt.close(fig)
        self.logger.info(f"Saved academic-grade visualization to: {output_path}")

    # ==================== Improved Textured Mode V2 ====================
    
    def _compute_non_overlapping_positions(
        self,
        positions: List[Tuple[int, int, int]],  # [(id, px, py), ...]
        min_distance: float = 25.0,
        max_iterations: int = 50,
    ) -> Dict[int, Tuple[int, int]]:
        """
        计算防重叠的节点位置
        
        使用简单的斥力算法将重叠的节点推开
        
        Args:
            positions: [(id, px, py), ...] 节点 ID 和原始像素位置
            min_distance: 节点之间的最小距离（像素）
            max_iterations: 最大迭代次数
            
        Returns:
            {id: (new_px, new_py)} 调整后的位置
        """
        if not positions:
            return {}
        
        # 转换为可变数组
        pos_dict = {p[0]: [float(p[1]), float(p[2])] for p in positions}
        ids = list(pos_dict.keys())
        
        for _ in range(max_iterations):
            moved = False
            for i, id_i in enumerate(ids):
                for j, id_j in enumerate(ids):
                    if i >= j:
                        continue
                    
                    dx = pos_dict[id_j][0] - pos_dict[id_i][0]
                    dy = pos_dict[id_j][1] - pos_dict[id_i][1]
                    dist = np.sqrt(dx**2 + dy**2)
                    
                    if dist < min_distance and dist > 0:
                        # 计算斥力方向
                        overlap = min_distance - dist
                        nx = dx / dist
                        ny = dy / dist
                        
                        # 推开两个节点
                        push = overlap / 2 + 1
                        pos_dict[id_i][0] -= nx * push
                        pos_dict[id_i][1] -= ny * push
                        pos_dict[id_j][0] += nx * push
                        pos_dict[id_j][1] += ny * push
                        moved = True
            
            if not moved:
                break
        
        return {k: (int(v[0]), int(v[1])) for k, v in pos_dict.items()}
    
    def visualize_textured_v2(
        self,
        top_down_map: np.ndarray,
        output_path: str,
        agent_position: Optional[np.ndarray] = None,
        agent_heading: Optional[float] = None,
        object_positions: Optional[Dict[int, np.ndarray]] = None,
        object_classes: Optional[Dict[int, str]] = None,
        trajectory: Optional[np.ndarray] = None,
        snapshot_positions: Optional[List[np.ndarray]] = None,
        snapshot_connections: Optional[List[Tuple[int, int]]] = None,
        object_to_snapshot: Optional[Dict[int, int]] = None,
        target_object_id: Optional[int] = None,
        exploration_masks: Optional[Dict[str, np.ndarray]] = None,
        title: str = "Explicit Memory Graph",
        show_object_labels: bool = False,
        show_legend: bool = True,
        show_exploration_legend: bool = True,
        prevent_overlap: bool = True,
        min_node_distance: float = 30.0,
        figsize: Tuple[float, float] = (16, 14),
        dpi: int = 300,
    ):
        """
        改进版学术俯视可视化 V2
        
        【改进点】
        1. 防重叠：节点位置自动调整，避免重叠
        2. 无蓝色光晕：移除目标周围的多层光晕效果
        3. 探索状态图例：可选显示已探索/未探索颜色说明
        4. 更紧凑的布局：减少留白
        5. 自适应节点大小：根据地图实际占用区域调整
        
        Args:
            top_down_map: 外部提供的高质量 RGB 俯视底图 (H, W, 3)
            output_path: 输出路径（.png）
            agent_position: 机器人当前 3D 位置 (3,) [x, y, z]
            agent_heading: 机器人朝向角度（弧度）
            object_positions: 物体位置 {obj_id: (3,) position}
            object_classes: 物体类别 {obj_id: class_name}
            trajectory: 机器人轨迹 (N, 3)
            snapshot_positions: Snapshot 位置列表 [(3,), ...]
            snapshot_connections: Snapshot 之间的连接 [(i, j), ...]
            object_to_snapshot: 物体到 Snapshot 的关联 {obj_id: snapshot_idx}
            target_object_id: 目标物体 ID
            exploration_masks: 探索状态掩码字典（来自 get_exploration_aware_topdown_map）
            title: 图像标题
            show_object_labels: 是否显示物体类别标签
            show_legend: 是否显示图例
            show_exploration_legend: 是否显示探索状态图例
            prevent_overlap: 是否启用防重叠
            min_node_distance: 防重叠时节点最小距离（像素）
            figsize: 图像尺寸
            dpi: 输出 DPI
        """
        if top_down_map is None or top_down_map.size == 0:
            self.logger.warning("No top_down_map provided for textured mode")
            return

        # ==================== 配色方案 ====================
        COLOR_IMAGE_NODE = '#4A90D9'        # 蓝色 - 图像节点
        COLOR_IMAGE_NODE_EDGE = '#2C5282'
        COLOR_OBJECT_NODE = '#FF69B4'       # 粉色 - 对象节点
        COLOR_OBJECT_NODE_EDGE = '#DB7093'
        COLOR_TRAJECTORY = '#87CEEB'        # 浅蓝色 - 轨迹线
        COLOR_ASSOCIATION = '#FFB6C1'       # 浅粉色 - 关联线
        COLOR_AGENT = '#32CD32'             # 绿色 - 当前位置
        COLOR_AGENT_EDGE = '#228B22'
        COLOR_TARGET = '#DC143C'            # 红色 - 目标
        
        # 探索状态颜色（与底图生成一致）
        COLOR_EXPLORED = '#FFFFFF'          # 白色 - 已探索
        COLOR_OBSERVED = '#EBEBEB'          # 浅灰 - 已观测
        COLOR_UNEXPLORED = '#C8C8C8'        # 深灰 - 未探索

        # ==================== 图像设置 ====================
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        fig.patch.set_facecolor('white')
        
        # 绘制底图
        ax.imshow(top_down_map, origin="upper", interpolation='nearest')  # 使用 nearest 保持清晰
        map_shape = top_down_map.shape[:2]
        
        # 动态计算节点大小
        map_diag = np.sqrt(map_shape[0]**2 + map_shape[1]**2)
        base_radius = max(8, min(15, map_diag / 100))
        trajectory_width = base_radius * 0.8
        
        # ==================== 收集所有节点位置用于防重叠 ====================
        all_node_positions = []  # [(id, px, py), ...]
        
        # Snapshot 节点
        snapshot_pixels = {}
        if snapshot_positions is not None:
            for i, pos in enumerate(snapshot_positions):
                px, py = self.world_to_pixel(pos, map_shape)
                snapshot_pixels[f"snap_{i}"] = (px, py)
                all_node_positions.append((f"snap_{i}", px, py))
        
        # 对象节点
        background_classes = {'wall', 'floor', 'ceiling', 'paneling', 'banner', 'misc', 'unknown'}
        object_pixels = {}
        if object_positions:
            for obj_id, pos in object_positions.items():
                if target_object_id is not None and obj_id == target_object_id:
                    continue
                class_name = object_classes.get(obj_id, '') if object_classes else ''
                if class_name.lower() in background_classes:
                    continue
                px, py = self.world_to_pixel(pos, map_shape)
                object_pixels[f"obj_{obj_id}"] = (px, py)
                all_node_positions.append((f"obj_{obj_id}", px, py))
        
        # 目标节点
        target_pixel = None
        if target_object_id is not None and object_positions and target_object_id in object_positions:
            pos = object_positions[target_object_id]
            px, py = self.world_to_pixel(pos, map_shape)
            target_pixel = (px, py)
            all_node_positions.append(("target", px, py))
        
        # Agent 位置
        agent_pixel = None
        if agent_position is not None:
            px, py = self.world_to_pixel(agent_position, map_shape)
            agent_pixel = (px, py)
            all_node_positions.append(("agent", px, py))
        
        # ==================== 防重叠处理 ====================
        if prevent_overlap and len(all_node_positions) > 1:
            adjusted_positions = self._compute_non_overlapping_positions(
                all_node_positions, 
                min_distance=min_node_distance
            )
            
            # 更新位置
            for key, new_pos in adjusted_positions.items():
                if key.startswith("snap_"):
                    snapshot_pixels[key] = new_pos
                elif key.startswith("obj_"):
                    object_pixels[key] = new_pos
                elif key == "target":
                    target_pixel = new_pos
                elif key == "agent":
                    agent_pixel = new_pos
        
        # ==================== Layer 1: 轨迹线 ====================
        if snapshot_positions is not None and len(snapshot_positions) > 1:
            snap_pos_list = [snapshot_pixels.get(f"snap_{i}", self.world_to_pixel(snapshot_positions[i], map_shape)) 
                           for i in range(len(snapshot_positions))]
            snap_pos_arr = np.array(snap_pos_list)
            
            if snapshot_connections is not None:
                for (i, j) in snapshot_connections:
                    if i < len(snap_pos_arr) and j < len(snap_pos_arr):
                        ax.plot(
                            [snap_pos_arr[i, 0], snap_pos_arr[j, 0]],
                            [snap_pos_arr[i, 1], snap_pos_arr[j, 1]],
                            color=COLOR_TRAJECTORY,
                            linewidth=trajectory_width,
                            alpha=0.6,
                            solid_capstyle='round',
                            zorder=2
                        )
            else:
                ax.plot(
                    snap_pos_arr[:, 0], snap_pos_arr[:, 1],
                    color=COLOR_TRAJECTORY,
                    linewidth=trajectory_width,
                    alpha=0.6,
                    solid_capstyle='round',
                    zorder=2
                )

        # ==================== Layer 2: 关联线 ====================
        if object_positions and object_to_snapshot and snapshot_positions:
            for obj_id, snapshot_idx in object_to_snapshot.items():
                if target_object_id is not None and obj_id == target_object_id:
                    continue
                obj_key = f"obj_{obj_id}"
                snap_key = f"snap_{snapshot_idx}"
                if obj_key in object_pixels and snap_key in snapshot_pixels:
                    obj_px, obj_py = object_pixels[obj_key]
                    snap_px, snap_py = snapshot_pixels[snap_key]
                    ax.plot(
                        [snap_px, obj_px], [snap_py, obj_py],
                        color=COLOR_ASSOCIATION,
                        linewidth=1.5,
                        alpha=0.5,
                        solid_capstyle='round',
                        zorder=3
                    )

        # ==================== Layer 3: Snapshot 节点 ====================
        if snapshot_positions is not None:
            for i, _ in enumerate(snapshot_positions):
                key = f"snap_{i}"
                if key in snapshot_pixels:
                    px, py = snapshot_pixels[key]
                    circle = plt.Circle(
                        (px, py), 
                        radius=base_radius,
                        facecolor=COLOR_IMAGE_NODE,
                        edgecolor=COLOR_IMAGE_NODE_EDGE,
                        linewidth=2.0,
                        alpha=0.9,
                        zorder=10
                    )
                    ax.add_patch(circle)

        # ==================== Layer 4: 对象节点 ====================
        if object_positions:
            for obj_id, pos in object_positions.items():
                if target_object_id is not None and obj_id == target_object_id:
                    continue
                class_name = object_classes.get(obj_id, '') if object_classes else ''
                if class_name.lower() in background_classes:
                    continue
                
                obj_key = f"obj_{obj_id}"
                if obj_key in object_pixels:
                    px, py = object_pixels[obj_key]
                    triangle_size = base_radius * 0.8
                    triangle = plt.Polygon(
                        [
                            (px, py - triangle_size * 1.1),
                            (px - triangle_size, py + triangle_size * 0.55),
                            (px + triangle_size, py + triangle_size * 0.55),
                        ],
                        facecolor=COLOR_OBJECT_NODE,
                        edgecolor=COLOR_OBJECT_NODE_EDGE,
                        linewidth=1.5,
                        alpha=0.85,
                        zorder=15
                    )
                    ax.add_patch(triangle)
                    
                    if show_object_labels and class_name:
                        ax.text(
                            px, py + triangle_size * 1.8,
                            class_name,
                            fontsize=max(5, base_radius * 0.4),
                            color='#2C3E50',
                            fontweight='semibold',
                            ha='center',
                            va='top',
                            bbox=dict(
                                boxstyle='round,pad=0.1',
                                facecolor='white',
                                alpha=0.7,
                                edgecolor='#BDC3C7',
                                linewidth=0.5
                            ),
                            zorder=16
                        )

        # ==================== Layer 5: 目标节点（无光晕）====================
        if target_pixel is not None:
            px, py = target_pixel
            # 简单的红色圆圈，无多层光晕
            target_circle = plt.Circle(
                (px, py),
                radius=base_radius * 1.3,
                facecolor=COLOR_TARGET,
                edgecolor='white',
                linewidth=3.0,
                alpha=0.95,
                zorder=19
            )
            ax.add_patch(target_circle)
            
            if object_classes and target_object_id in object_classes:
                target_name = object_classes[target_object_id]
                ax.text(
                    px, py - base_radius * 2.5,
                    f"Goal: {target_name}",
                    fontsize=max(7, base_radius * 0.5),
                    color='white',
                    fontweight='bold',
                    ha='center',
                    bbox=dict(
                        boxstyle='round,pad=0.3',
                        facecolor=COLOR_TARGET,
                        alpha=0.9,
                        edgecolor='white',
                        linewidth=1.5
                    ),
                    zorder=20
                )

        # ==================== Layer 6: Agent 位置 ====================
        if agent_pixel is not None:
            px, py = agent_pixel
            agent_circle = plt.Circle(
                (px, py),
                radius=base_radius * 1.2,
                facecolor=COLOR_AGENT,
                edgecolor=COLOR_AGENT_EDGE,
                linewidth=2.5,
                alpha=0.95,
                zorder=25
            )
            ax.add_patch(agent_circle)
            
            if agent_heading is not None:
                arrow_len = base_radius * 2.2
                dx = arrow_len * np.sin(agent_heading)
                dy = -arrow_len * np.cos(agent_heading)
                ax.arrow(
                    px, py, dx, dy,
                    head_width=base_radius * 0.6,
                    head_length=base_radius * 0.4,
                    fc='#FFD700',
                    ec='#2C3E50',
                    linewidth=1.5,
                    zorder=26
                )

        # ==================== 图像属性 ====================
        ax.set_xlim(0, map_shape[1])
        ax.set_ylim(map_shape[0], 0)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # ==================== 图例 ====================
        if show_legend:
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', 
                           markerfacecolor=COLOR_IMAGE_NODE, markersize=12, 
                           label='Image Node', markeredgecolor=COLOR_IMAGE_NODE_EDGE, 
                           markeredgewidth=1.5, linestyle='None'),
                plt.Line2D([0], [0], marker='^', color='w', 
                           markerfacecolor=COLOR_OBJECT_NODE, markersize=12, 
                           label='Object Node', markeredgecolor=COLOR_OBJECT_NODE_EDGE, 
                           markeredgewidth=1.5, linestyle='None'),
                plt.Line2D([0], [0], marker='o', color='w', 
                           markerfacecolor=COLOR_AGENT, markersize=12, 
                           label='Current Position', markeredgecolor=COLOR_AGENT_EDGE, 
                           markeredgewidth=1.5, linestyle='None'),
                plt.Line2D([0], [0], color=COLOR_TRAJECTORY, 
                           linewidth=6, alpha=0.6, label='Trajectory'),
            ]
            
            if target_object_id is not None:
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor=COLOR_TARGET, markersize=14, 
                               label='Goal', markeredgecolor='white', 
                               markeredgewidth=2, linestyle='None')
                )
            
            # 探索状态图例
            if show_exploration_legend:
                legend_elements.extend([
                    plt.Line2D([0], [0], marker='s', color='w', 
                               markerfacecolor=COLOR_EXPLORED, markersize=12, 
                               label='Explored', markeredgecolor='#282828', 
                               markeredgewidth=1, linestyle='None'),
                    plt.Line2D([0], [0], marker='s', color='w', 
                               markerfacecolor=COLOR_OBSERVED, markersize=12, 
                               label='Observed', markeredgecolor='#969696', 
                               markeredgewidth=1, linestyle='None'),
                    plt.Line2D([0], [0], marker='s', color='w', 
                               markerfacecolor=COLOR_UNEXPLORED, markersize=12, 
                               label='Unexplored', markeredgecolor='#A0A0A0', 
                               markeredgewidth=1, linestyle='None'),
                ])
            
            legend = ax.legend(
                handles=legend_elements,
                loc='upper right',
                fontsize=max(8, base_radius * 0.6),
                framealpha=0.9,
                fancybox=True,
                frameon=True,
                edgecolor='#CCCCCC',
                borderpad=0.6,
                labelspacing=0.6,
            )
            legend.get_frame().set_linewidth(1.0)

        # ==================== 保存 ====================
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.tight_layout(pad=0.2)
        plt.savefig(
            output_path, 
            dpi=dpi, 
            bbox_inches='tight', 
            facecolor='white', 
            edgecolor='none',
            pad_inches=0.05
        )
        plt.close(fig)
        self.logger.info(f"Saved improved visualization V2 to: {output_path}")

    # ==================== Topology Mode (原有功能) ====================

    def visualize_hierarchical_graph(
        self,
        regions: Dict[int, Any],
        scene_objects: Dict[int, Dict[str, Any]],
        obj_to_region: Dict[int, int],
        floor_id: str,
        output_path: str,
        decision_history: Optional[Dict[int, List[str]]] = None,
        bg_image: Optional[np.ndarray] = None,
    ):
        """
        Topology Mode: 语义拓扑图（2D 俯视图）
        
        特点：
        - 可选 Habitat top_down_map 作为背景
        - 绘制房间→物体虚线连线
        - 根据 decision_history 高亮冲突节点
        - 强制 axis('equal') 等比例坐标
        
        Args:
            regions: 区域字典 {region_id: RegionNode}
            scene_objects: 物体字典 {obj_id: {class_name, pcd, bbox, ...}}
            obj_to_region: 物体到区域的映射 {obj_id: region_id}
            floor_id: 楼层标识
            output_path: 输出路径（.png）
            decision_history: 决策历史 {obj_id: [decision1, decision2, ...]}
            bg_image: 可选的背景图像 (H, W, 3) RGB，从 Habitat 仿真器获取的 top_down_map
        """
        if not scene_objects:
            self.logger.warning("No objects to visualize")
            return

        fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
        
        # 如果提供了背景图，先绘制背景
        if bg_image is not None and bg_image.size > 0:
            ax.imshow(bg_image, origin='upper', alpha=0.6, interpolation='bilinear')
            # 使用图像坐标系统
            use_pixel_coords = True
            map_shape = bg_image.shape[:2]
        else:
            use_pixel_coords = False
            map_shape = None
        
        # 计算中心点（2D 投影：XZ 平面）
        room_centroids_2d = self._compute_room_centroids_2d(regions, floor_id)
        obj_centroids_2d = self._compute_object_centroids_2d(scene_objects)
        
        if not room_centroids_2d:
            self.logger.warning("No room centroids computed")
            plt.close(fig)
            return
        
        # 如果使用像素坐标，转换所有坐标
        if use_pixel_coords and map_shape is not None:
            room_centroids_2d = {
                k: self.world_to_pixel(np.array([v[0], 0, v[1]]), map_shape)
                for k, v in room_centroids_2d.items()
            }
            obj_centroids_2d = {
                k: self.world_to_pixel(np.array([v[0], 0, v[1]]), map_shape)
                for k, v in obj_centroids_2d.items()
            }
        
        # 1) 绘制房间节点（蓝色圆圈）
        node_radius = 15 if use_pixel_coords else 0.3
        for room_id_str, centroid_2d in room_centroids_2d.items():
            circle = Circle(
                centroid_2d, 
                radius=node_radius, 
                facecolor='blue', 
                edgecolor='darkblue',
                alpha=0.6,
                linewidth=2,
                zorder=10
            )
            ax.add_patch(circle)
            text_offset = node_radius * 1.5 if use_pixel_coords else 0.5
            ax.text(
                centroid_2d[0], centroid_2d[1] + text_offset, 
                f"R{room_id_str.split('_')[-1]}", 
                ha='center', va='bottom',
                fontsize=8, fontweight='bold',
                color='darkblue'
            )
        
        # 2) 绘制物体节点 + 房间→物体连线
        for obj_id, obj in scene_objects.items():
            rid = obj_to_region.get(obj_id)
            if rid is None:
                continue
            
            room_id_str = f"{floor_id}_{rid}"
            if room_id_str not in room_centroids_2d:
                continue
            
            obj_name = obj.get("class_name", "unknown")
            # 过滤墙、地板、天花板
            if any(
                substring in obj_name.lower()
                for substring in ["wall", "floor", "ceiling", "paneling", "banner"]
            ):
                continue
            
            obj_centroid_2d = obj_centroids_2d.get(str(obj_id))
            if obj_centroid_2d is None:
                continue
            
            # 判断是否为冲突节点
            obj_color, obj_marker = self._get_object_style(obj_id, decision_history)
            
            # 绘制房间→物体连线（虚线）
            room_center = room_centroids_2d[room_id_str]
            ax.plot(
                [room_center[0], obj_centroid_2d[0]],
                [room_center[1], obj_centroid_2d[1]],
                color='gray',
                linestyle='--',
                linewidth=1.0,
                alpha=0.5,
                zorder=1
            )
            
            # 绘制物体节点
            ax.scatter(
                obj_centroid_2d[0], 
                obj_centroid_2d[1],
                c=obj_color,
                marker=obj_marker,
                s=80,
                edgecolors='black',
                linewidths=1.5,
                alpha=0.8,
                zorder=5
            )
            
            # 标注物体名称
            ax.text(
                obj_centroid_2d[0], 
                obj_centroid_2d[1] - 0.3,
                obj_name[:10],
                ha='center', va='top',
                fontsize=6,
                color=obj_color
            )
        
        # 强制等比例坐标轴
        ax.axis('equal')
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Z (m)', fontsize=12)
        ax.set_title(
            f'Hierarchical Scene Graph - {floor_id}\n'
            f'(Objects: {len(scene_objects)}, Rooms: {len(room_centroids_2d)})',
            fontsize=14,
            fontweight='bold'
        )
        ax.grid(True, alpha=0.3, linestyle=':')
        
        # 添加图例
        self._add_legend(ax)
        
        # 保存
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        self.logger.info(f"Saved hierarchical graph to: {output_path}")

    def draw_decision_heatmap(
        self,
        decision_events: List[Tuple[int, str, np.ndarray, Optional[np.ndarray]]],
        scene_objects: Dict[int, Dict[str, Any]],
        output_path: str,
        trajectory: Optional[np.ndarray] = None,
    ):
        """
        决策热力图（离散事件点）
        
        Args:
            decision_events: [(obj_id, decision, obj_pos_3d, robot_pos_3d), ...]
            scene_objects: 物体字典
            output_path: 输出路径
            trajectory: 机器人轨迹 (N, 3)
        """
        fig, ax = plt.subplots(figsize=(14, 10), dpi=150)
        
        # 1) 绘制物体位置（灰色背景点）
        for obj_id, obj in scene_objects.items():
            centroid_2d = self._get_object_centroid_2d(obj)
            if centroid_2d is not None:
                ax.scatter(
                    centroid_2d[0], centroid_2d[1],
                    c='lightgray',
                    marker='o',
                    s=30,
                    alpha=0.3,
                    zorder=1
                )
        
        # 2) 绘制机器人轨迹
        if trajectory is not None and len(trajectory) > 0:
            traj_2d = trajectory[:, [0, 2]]  # XZ 平面投影
            ax.plot(
                traj_2d[:, 0], traj_2d[:, 1],
                color='black',
                linestyle='-',
                linewidth=1.5,
                alpha=0.4,
                label='Robot Trajectory',
                zorder=2
            )
        
        # 3) 绘制决策事件点
        decision_counts = defaultdict(int)
        
        for obj_id, decision, obj_pos_3d, robot_pos_3d in decision_events:
            if decision == "KEEP":
                if robot_pos_3d is not None:
                    pos_2d = robot_pos_3d[[0, 2]]
                else:
                    pos_2d = obj_pos_3d[[0, 2]]
                marker, color, size = 'x', 'red', 150
            elif decision == "REPLACE":
                pos_2d = obj_pos_3d[[0, 2]]
                marker, color, size = '^', 'yellow', 120
            elif decision == "SPLIT":
                pos_2d = obj_pos_3d[[0, 2]]
                marker, color, size = 's', 'orange', 100
            else:
                continue
            
            ax.scatter(
                pos_2d[0], pos_2d[1],
                c=color,
                marker=marker,
                s=size,
                edgecolors='black',
                linewidths=1.5,
                alpha=0.9,
                zorder=10
            )
            decision_counts[decision] += 1
        
        # 设置图像属性
        ax.axis('equal')
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Z (m)', fontsize=12)
        ax.set_title(
            f'Decision Heatmap\n'
            f'KEEP: {decision_counts["KEEP"]}, REPLACE: {decision_counts["REPLACE"]}, SPLIT: {decision_counts["SPLIT"]}',
            fontsize=14, fontweight='bold'
        )
        ax.grid(True, alpha=0.3, linestyle=':')
        
        # 图例
        legend_elements = [
            Line2D([0], [0], marker='x', color='w', markerfacecolor='red', 
                   markersize=10, label='KEEP', markeredgecolor='black'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='yellow', 
                   markersize=10, label='REPLACE', markeredgecolor='black'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', 
                   markersize=8, label='SPLIT', markeredgecolor='black'),
        ]
        if trajectory is not None:
            legend_elements.append(
                Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='Trajectory')
            )
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # 保存
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        self.logger.info(f"Saved decision heatmap to: {output_path}")

    def visualize_conflict_statistics(
        self,
        decision_history: Dict[int, List[str]],
        output_path: str,
    ):
        """绘制冲突决策统计图（柱状图 + 饼图）"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
        
        # 统计决策类型
        decision_counts = defaultdict(int)
        for obj_id, decisions in decision_history.items():
            for decision in decisions:
                decision_counts[decision] += 1
        
        if not decision_counts:
            decision_counts["NONE"] = 0
        
        # 1) 柱状图
        decisions = list(decision_counts.keys())
        counts = list(decision_counts.values())
        colors = [self.decision_colors.get(d, 'gray') for d in decisions]
        
        ax1.bar(decisions, counts, color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Decision Type Distribution', fontsize=14, fontweight='bold')
        ax1.grid(True, axis='y', alpha=0.3)
        
        max_count = max(counts) if counts else 1
        for i, (d, c) in enumerate(zip(decisions, counts)):
            ax1.text(i, c + max_count * 0.02, str(c), 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 2) 饼图
        conflict_objects = sum(1 for decisions in decision_history.values() 
                              if any(d in ["SPLIT", "REPLACE", "KEEP"] for d in decisions))
        normal_objects = len(decision_history) - conflict_objects
        
        if conflict_objects + normal_objects == 0:
            conflict_objects = 0
            normal_objects = 1
        
        ax2.pie(
            [conflict_objects, normal_objects],
            labels=['Conflict', 'Normal'],
            colors=['red', 'lightgray'],
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'}
        )
        ax2.set_title('Object Conflict Ratio', fontsize=14, fontweight='bold')
        
        # 保存
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        self.logger.info(f"Saved conflict statistics to: {output_path}")

    # ==================== 统一接口 ====================

    def visualize(
        self,
        mode: VisualizationMode,
        output_path: str,
        *,
        # Abstract mode params
        regions: Optional[Dict[int, Any]] = None,
        # Textured mode params
        top_down_map: Optional[np.ndarray] = None,
        agent_position: Optional[np.ndarray] = None,
        agent_heading: Optional[float] = None,
        object_positions: Optional[Dict[int, np.ndarray]] = None,
        trajectory: Optional[np.ndarray] = None,
        # Topology mode params
        scene_objects: Optional[Dict[int, Dict[str, Any]]] = None,
        obj_to_region: Optional[Dict[int, int]] = None,
        decision_history: Optional[Dict[int, List[str]]] = None,
        floor_id: str = "0",
        title: str = "",
    ):
        """
        统一可视化接口
        
        Args:
            mode: 可视化模式 (ABSTRACT / TEXTURED / TOPOLOGY)
            output_path: 输出路径
            其他参数根据模式选择性提供
        """
        if mode == VisualizationMode.TEXTURED:
            self.visualize_textured(
                top_down_map=top_down_map,
                output_path=output_path,
                agent_position=agent_position,
                agent_heading=agent_heading,
                object_positions=object_positions,
                trajectory=trajectory,
                title=title or "Textured Top-Down View",
            )
        
        elif mode == VisualizationMode.TOPOLOGY:
            self.visualize_hierarchical_graph(
                regions=regions or {},
                scene_objects=scene_objects or {},
                obj_to_region=obj_to_region or {},
                floor_id=floor_id,
                output_path=output_path,
                decision_history=decision_history,
            )
        
        else:
            self.logger.error(f"Unknown visualization mode: {mode}")

    # ==================== 辅助方法 ====================

    def _compute_room_centroids_2d(
        self, regions: Dict[int, Any], floor_id: str
    ) -> Dict[str, np.ndarray]:
        """计算房间中心点（2D 投影 XZ 平面）"""
        centroids = {}
        for rid, rn in regions.items():
            if rn.mask is not None:
                coords = np.argwhere(rn.mask)
                if len(coords) > 0:
                    coords_world = coords * self.voxel_size
                    center_3d = coords_world.mean(axis=0)
                    center_2d = center_3d[[0, 2]] if len(center_3d) >= 3 else center_3d[:2]
                    centroids[f"{floor_id}_{rid}"] = center_2d
        return centroids

    def _compute_object_centroids_2d(
        self, scene_objects: Dict[int, Dict[str, Any]]
    ) -> Dict[str, np.ndarray]:
        """计算物体中心点（2D 投影）"""
        centroids = {}
        for obj_id, obj in scene_objects.items():
            centroid_2d = self._get_object_centroid_2d(obj)
            if centroid_2d is not None:
                centroids[str(obj_id)] = centroid_2d
        return centroids

    def _get_object_centroid_2d(self, obj: Dict[str, Any]) -> Optional[np.ndarray]:
        """获取单个物体的 2D 中心点（XZ 平面）"""