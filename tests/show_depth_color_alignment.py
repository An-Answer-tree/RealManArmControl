import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读图
rgb   = cv2.imread('/home/szliutong/Project/RealManArmControl/saved_images/RBG_images/RBG_1280x720_58747733.png')
depth = cv2.imread('/home/szliutong/Project/RealManArmControl/saved_images/depth_images/depth_1280x720_58747733.png', cv2.IMREAD_UNCHANGED)  # uint16

# 2. 转为 8 位灰度
#    方法 A：normalize → uint8
depth8 = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
depth8 = depth8.astype(np.uint8)

#    （也可以手动除以 256，但 normalize 更保险）
#    depth8 = (depth >> 8).astype(np.uint8)

# 3. Canny 边缘检测
edges = cv2.Canny(depth8, 50, 150)

# 4. 在 RGB 图上用红色高亮深度边缘
overlay = rgb.copy()
overlay[edges > 0] = (0, 0, 255)  # BGR

# 5. 可视化
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title('RGB')
plt.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)); plt.axis('off')
plt.subplot(1,3,2)
plt.title('Depth → 8bit')
plt.imshow(depth8, cmap='gray'); plt.axis('off')
plt.subplot(1,3,3)
plt.title('Overlay Edges')
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)); plt.axis('off')
plt.tight_layout()
plt.show()
