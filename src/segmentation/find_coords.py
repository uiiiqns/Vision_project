import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

img = Image.open("/home/user/Vision_project/data/processed/0001.jpg")
arr = np.array(img)

fig, ax = plt.subplots(1, 1, figsize=(20, 13))
ax.imshow(arr)
ax.set_title("during_fire - 산불 지역 좌표 확인")

# 격자 표시
for x in range(0, arr.shape[1], 500):
    ax.axvline(x, color='white', alpha=0.3, linewidth=0.5)
    ax.text(x, 50, str(x), color='white', fontsize=8)
for y in range(0, arr.shape[0], 500):
    ax.axhline(y, color='white', alpha=0.3, linewidth=0.5)
    ax.text(50, y, str(y), color='white', fontsize=8)

plt.savefig("/home/user/Vision_project/results/during_fire_grid.jpg", dpi=100, bbox_inches='tight')
print("격자 이미지 저장 완료!")
