#!/usr/bin/env python3
"""
grasp_anything x GG-CNN adapter
===============================

grasp_anything_2d 内含 locateanything (VLM 定位)，
本适配器复用 locateanything 的集成逻辑，增加抓取接触点分析。

流水线:
  1. locateanything 定位目标 → bbox
  2. GG-CNN 在深度图上预测抓取位姿
  3. 可选: 生成抓取接触点 (grasp contact points)

用法:
  python integrations/ggcnn/run_ggcnn.py \
    --image scene.jpg --depth depth.tiff \
    --query "red mug" --locate-url http://127.0.0.1:8000
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GGCNN_ROOT = Path(
    os.environ.get("GGCNN_ROOT", PROJECT_ROOT.parent / "ggcnn")
).expanduser()
sys.path.insert(0, str(GGCNN_ROOT))

from ggcnn_bridge import GGCNNBridge  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def locate_object(image_path, query, locate_url, mode="ground_single"):
    """调用 locateanything API"""
    import httpx
    with open(image_path, "rb") as f:
        resp = httpx.post(
            f"{locate_url}/v1/locate",
            files={"image": (os.path.basename(image_path), f, "image/jpeg")},
            data={"query": query, "mode": mode, "generation_mode": "hybrid"},
            timeout=300, trust_env=False,
        )
    resp.raise_for_status()
    return resp.json()


def grasp_to_contact_points(g, depth_shape):
    """从抓取参数生成两个接触点 (gripper 两侧)"""
    cx, cy = g['center']
    ang = g['angle_rad']
    half_w = g['width'] / 2.0
    xo, yo = np.cos(ang), np.sin(ang)
    # 接触点 = center ± (half_w * perpendicular)
    p1 = [cx - half_w * yo, cy + half_w * xo]
    p2 = [cx + half_w * yo, cy - half_w * xo]
    return [p1, p2]


def main():
    p = argparse.ArgumentParser(description='grasp_anything x GG-CNN')
    p.add_argument('--image', required=True)
    p.add_argument('--depth', required=True)
    p.add_argument('--query', required=True)
    p.add_argument('--locate-url', default='http://127.0.0.1:8000')
    p.add_argument('--locate-mode', default='ground_single')
    p.add_argument('--network', default='ggcnn', choices=['ggcnn', 'ggcnn2'])
    p.add_argument(
        '--weights',
        default=str(GGCNN_ROOT / 'weights' / 'ggcnn_epoch_23_cornell_statedict.pt'),
    )
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--n-grasps', type=int, default=1)
    p.add_argument('--save', default=None)
    p.add_argument('--output-json', default=None, help='保存结果为 JSON')
    args = p.parse_args()

    # 定位
    logger.info(f"调用 locateanything: query='{args.query}'")
    loc_result = locate_object(
        args.image, args.query, args.locate_url, args.locate_mode
    )

    boxes = []
    if "boxes" in loc_result:
        boxes = loc_result["boxes"]
    elif "detections" in loc_result:
        for det in loc_result["detections"]:
            if "bbox" in det:
                boxes.append(det["bbox"])

    if not boxes:
        logger.warning("未返回 bbox, 使用整图")
        boxes = [[0, 0, -1, -1]]

    from imageio.v2 import imread
    depth = imread(args.depth).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]

    bridge = GGCNNBridge(
        mode='direct', ggcnn_root=GGCNN_ROOT, network=args.network,
        weights=args.weights, device=args.device,
    )

    all_grasps = []
    for box in boxes:
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        if x2 < 0:
            x1, y1, x2, y2 = 0, 0, depth.shape[1], depth.shape[0]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(depth.shape[1], x2), min(depth.shape[0], y2)
        if x2 - x1 < 10 or y2 - y1 < 10:
            continue
        depth_crop = depth[y1:y2, x1:x2]
        grasps = bridge.predict(depth_crop, n_grasps=args.n_grasps)
        for g in grasps:
            g['center'][0] += x1
            g['center'][1] += y1
            g['bbox'] = [x1, y1, x2, y2]
            g['query'] = args.query
            g['contact_points'] = grasp_to_contact_points(g, depth.shape)
            all_grasps.append(g)

    logger.info(f"总计 {len(all_grasps)} 个抓取 (含接触点)")
    for g in all_grasps:
        logger.info(f"  center=({g['center'][0]:.0f},{g['center'][1]:.0f}) "
                     f"angle={g['angle_deg']:.1f}° contacts={g['contact_points']}")

    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump({"query": args.query, "grasps": all_grasps}, f, indent=2)
        logger.info(f"结果已保存到 {args.output_json}")

    if args.save:
        import matplotlib
        matplotlib.use('Agg')
        import cv2
        import matplotlib.pyplot as plt
        img = cv2.imread(args.image)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            fig, ax = plt.subplots(1, 1, figsize=(12, 9))
            ax.imshow(img)
            for g in all_grasps:
                cx, cy = g['center']
                ang = g['angle_rad']
                width, length = g['width'], g['length']
                xo, yo = np.cos(ang), np.sin(ang)
                rect = plt.Rectangle(
                    (
                        cx - length / 2 * yo - width / 2 * xo,
                        cy - length / 2 * xo + width / 2 * yo,
                    ),
                    length,
                    width,
                    angle=np.degrees(ang),
                    fill=False,
                    color='magenta',
                    linewidth=2,
                )
                ax.add_patch(rect)
                # 画接触点
                for cp in g['contact_points']:
                    ax.plot(cp[0], cp[1], 'r.', markersize=10)
            ax.set_title(f'grasp_anything + GG-CNN: "{args.query}"')
            ax.axis('off')
            plt.savefig(args.save, dpi=150, bbox_inches='tight')
            logger.info(f"可视化已保存到 {args.save}")


if __name__ == '__main__':
    main()
