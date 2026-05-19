import ee
import geemap
import os

ee.Initialize(project='wildfire-project-496714')

# LA 산불 지역 ROI (팰리세이즈 산불 중심)
roi = ee.Geometry.Rectangle([-118.6, 34.0, -118.0, 34.4])

def get_sentinel2(start_date, end_date, cloud_pct=20):
    return (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(roi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_pct))
            .select(['B2','B3','B4','B8','B11','B12']))

def add_indices(image):
    nbr = image.normalizedDifference(['B8', 'B12']).rename('NBR').toFloat()
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI').toFloat()
    return image.toFloat().addBands([nbr, ndvi])

# 산불 전/중/후 시계열
pre_fire  = get_sentinel2('2024-11-01', '2025-01-06').map(add_indices)
during_fire = get_sentinel2('2025-01-07', '2025-01-20').map(add_indices)
post_fire = get_sentinel2('2025-01-21', '2025-02-28').map(add_indices)

print(f"산불 전 이미지 수: {pre_fire.size().getInfo()}")
print(f"산불 중 이미지 수: {during_fire.size().getInfo()}")
print(f"산불 후 이미지 수: {post_fire.size().getInfo()}")
# 각 시기별 중앙값 합성 이미지 생성
pre_composite = pre_fire.median().clip(roi)
during_composite = during_fire.median().clip(roi)
post_composite = post_fire.median().clip(roi)

# 다운로드
output_dir = '../../data/raw'
os.makedirs(output_dir, exist_ok=True)

for name, img in [('pre_fire', pre_composite), 
                   ('during_fire', during_composite), 
                   ('post_fire', post_composite)]:
    task = ee.batch.Export.image.toDrive(
        image=img,
        description=name,
        folder='sentinel2_wildfire',
        fileNamePrefix=name,
        scale=10,
        region=roi,
        maxPixels=1e9
    )
    task.start()
    print(f'{name} export 시작!')

print('완료! https://code.earthengine.google.com/tasks 에서 진행상황 확인하세요')

