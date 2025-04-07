# %%
import cv2
from rasterio.features import rasterize
from datetime import datetime
import re
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box
import cv2  # 이미지 저장(예: png) 등을 위해 사용

# TODO: aoi에서 벗어난 영역(안 쓰는 부분에 대해서 mask를 얻어놔야 나중에 loss에 반영할 수 있음)

# 경로 설정 (사용자 환경에 맞게 변경)
aoi_path = '15/aoi/aoi.shp'
event_path = '15/event/event.shp'
hydroA_path = '15/hydro/hydroA.shp'
slc_folder = '15/slc_images'
output_folder = '15/output_patches'
os.makedirs(output_folder, exist_ok=True)
# shapefile 불러오기
aoi = gpd.read_file(aoi_path)
event = gpd.read_file(event_path)
hydroA = gpd.read_file(hydroA_path)
aoi_geoms = [feat["geometry"] for feat in aoi.__geo_interface__["features"]]

# SLC 이미지 파일 목록 읽기
# 파일 이름에 "post"가 포함된 것이 홍수 후 이미지라고 가정합니다.
slc_files = [os.path.join(slc_folder, f) for f in os.listdir(
    slc_folder) if f.lower().endswith('.tif')]


def get_acquisition_date(filename):
    """
    파일명에서 'YYYYMMDDTHHMMSS' 형식의 문자열을 찾아 datetime 객체로 반환.
    예: 'S1A_IW_SLC__1SDV_20210908T191537_20210908T191604_039594_04AE27_B611.tif'
    """
    match = re.search(r'(\d{8}T\d{6})', filename)
    if match:
        dt_str = match.group(1)
        return datetime.strptime(dt_str, '%Y%m%dT%H%M%S')
    return None


# 각 파일의 날짜 정보를 추출하여 (파일명, 날짜) 튜플 리스트로 구성
file_dates = []
for f in slc_files:
    acq_date = get_acquisition_date(f)
    if acq_date is not None:
        file_dates.append((f, acq_date))
    else:
        print(f"날짜 파싱 실패: {f}")

# 날짜 순으로 정렬 (오름차순)
file_dates.sort(key=lambda x: x[1])

if len(file_dates) != 3:
    print("경고: SLC 이미지 파일이 3개가 아닙니다!")
else:
    # 가장 최근 날짜를 post-event 이미지로, 나머지를 pre-event 이미지로 구분
    pre_files = [file_dates[0][0], file_dates[1][0]]
    post_files = [file_dates[2][0]]


def load_masked_image(filepath):
    # SLC 이미지를 AOI 영역으로 crop하는 함수
    # TODO 학습 시에 loss에서 마스킹된(AOI 영역 밖) 부분은 무시하도록 설정해야 함.
    with rasterio.open(filepath) as src:
        try:
            masked_img, masked_transform = mask(src, aoi_geoms, crop=True)
        except Exception as e:
            print(f"Error masking {filepath}: {e}")
            return None, None
    return masked_img, masked_transform


# 각 이미지를 dictionary에 저장: key는 'pre_1', 'pre_2', 'post'
images = {}
for idx, filepath in enumerate(pre_files):
    masked_img, masked_transform = load_masked_image(filepath)
    if masked_img is not None:
        images[f"pre_{idx+1}"] = {
            "data": masked_img,
            "transform": masked_transform,
            "filename": os.path.basename(filepath)
        }
for filepath in post_files:
    masked_img, masked_transform = load_masked_image(filepath)
    if masked_img is not None:
        images["post"] = {
            "data": masked_img,
            "transform": masked_transform,
            "filename": os.path.basename(filepath)
        }
# --- 3. Segmentation Mask 생성 (pixel-wise) ---
# post flood 이미지의 크기와 transform 사용
post_img = images["post"]["data"]
post_transform = images["post"]["transform"]
_, height, width = post_img.shape

# event (홍수) 영역 rasterize: label 1
flood_mask = rasterize(
    [(geom, 1) for geom in event.geometry],
    out_shape=(height, width),
    transform=post_transform,
    fill=0,
    all_touched=True,
    dtype=np.uint8
)
# hydroA (평상시 수역) rasterize: label 2 (단, 홍수 영역이 아닌 곳에만 적용)
water_mask = rasterize(
    [(geom, 1) for geom in hydroA.geometry],
    out_shape=(height, width),
    transform=post_transform,
    fill=0,
    all_touched=True,
    dtype=np.uint8
)
# 최종 segmentation mask: 홍수 영역 우선, 그 외 평상시 수역, 나머지는 land (0)
# 홍수 영역: 1, 평상시 수역: 2, land: 0
segmentation_mask = np.where(
    flood_mask == 1, 1, np.where(water_mask == 1, 2, 0))

# --- 4. 슬라이딩 윈도우를 통한 패치 및 segmentation mask 추출 ---

patch_size = 224
patch_records = []

# post 이미지 기준 패치 그리드 생성
for i in range(0, height, patch_size):
    for j in range(0, width, patch_size):
        if i + patch_size <= height and j + patch_size <= width:
            # post 이미지에서 패치 추출
            post_patch = post_img[:, i:i+patch_size, j:j+patch_size]
            # segmentation mask 패치 추출 (2D array)
            seg_patch = segmentation_mask[i:i+patch_size, j:j+patch_size]

            # 각 이미지(2개의 pre, 1개의 post)에서 동일한 좌표의 패치 추출
            patch_stack = []
            valid_patch = True
            for key, info in images.items():
                data = info["data"]
                transform = info["transform"]
                # post와 동일한 좌표를 사용하므로, world 좌표 계산 (post 기준)
                x_min, y_max = post_transform * (j, i)          # top-left
                x_max, y_min = post_transform * \
                    (j+patch_size, i+patch_size)  # bottom-right

                # 픽셀 인덱스 변환: 해당 이미지의 transform을 기준으로
                row_min, col_min = rasterio.transform.rowcol(
                    transform, x_min, y_max)
                row_max, col_max = rasterio.transform.rowcol(
                    transform, x_max, y_min)

                _, h, w = data.shape
                if row_min < 0 or col_min < 0 or row_max > h or col_max > w:
                    valid_patch = False
                    break
                patch_img = data[:, row_min:row_max, col_min:col_max]
                if patch_img.shape[1] != patch_size or patch_img.shape[2] != patch_size:
                    valid_patch = False
                    break
                patch_stack.append(patch_img)
            if not valid_patch:
                continue

            # 시간 순서대로 스택: [pre_1, pre_2, post] → shape: (3, bands, patch_size, patch_size)
            patch_stack = np.stack(patch_stack, axis=0)

            # 저장: 입력 patch stack와 대응하는 segmentation mask patch를 함께 저장
            patch_filename = f"patch_{i}_{j}.npz"
            patch_filepath = os.path.join(output_folder, patch_filename)
            np.savez(patch_filepath, patch=patch_stack, seg=seg_patch)

            # (옵션) post 이미지 패치 시각화 (PNG) - 3채널 이상일 경우
            if post_patch.shape[0] >= 3:
                patch_rgb = post_patch[:3]
                patch_rgb = (patch_rgb - patch_rgb.min()) / \
                    (patch_rgb.max() - patch_rgb.min() + 1e-6)
                patch_rgb = (patch_rgb * 255).astype(np.uint8)
                patch_rgb = np.transpose(patch_rgb, (1, 2, 0))
                cv2.imwrite(os.path.join(output_folder, f"patch_{i}_{j}.png"),
                            cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR))

            patch_records.append({
                "patch_file": patch_filename,
                "image_source_post": images["post"]["filename"],
                "patch_bounds": (j, i, j+patch_size, i+patch_size)
            })

# CSV 파일로 패치 정보 저장
df = pd.DataFrame(patch_records)
df.to_csv(os.path.join(output_folder, "patch_labels.csv"), index=False)
print("데이터셋 가공 완료. 생성된 패치 수:", len(patch_records))
# %%
df
