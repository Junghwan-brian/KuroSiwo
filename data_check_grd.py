# %%
import os
import rasterio.features
from rasterio.transform import array_bounds
import rasterio.mask
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import rasterio
import geopandas as gpd
'''
- MK0_MNA: No-data Mask
    → 유효하지 않거나 결측치인 픽셀을 표시하는 마스크입니다. (mask of valid pixels)
- MK0_MLU
    → mask of flooded/perm water pixels
    정답 값. 물이 있는 픽셀을 표시하는 마스크입니다.
- MK0_DEM: Digital Elevation Model
    → 해당 지역의 고도 정보를 나타내는 디지털 고도 모델입니다.
- MS: Master
    -> Flood image
- SL: Slave
    -> Non-flood image

class: No Water, Permanent Waters, Floods

Each tar contains a subset of the available events in Kuro Siwo. 
For each event there is a set of areas of interest e.g 00/, 01/, 02/, etc. 
The unlabeled component for each event is always listed in the 00/ directory. 
There is no annotation for these samples.

The rest of the AOIs contain the annotated component of Kuro Siwo. 
For example, for the AOI 01, of the event 1111002, we can find the sample with a hash code 8b15c563799b5aed819c26534d0d50f0 in 1111002/01/8b15c563799b5aed819c26534d0d50f0. 
The annotations are provided with the following naming convention: MK0_MLU_EVENTID_AOI_DATE.tif. 
In a similar fashion, we provide the DEM, the slope as well as the SAR captions.
'''
grid_dict_full = pd.read_pickle("pickle/grid_dict_full.pkl")
water_grid_dict = pd.read_pickle("pickle/grid_with_water_dict.pkl")
grid_dict = pd.read_pickle("pickle/grid_dict.pkl")
# %%
folder_path = "/home/junghwan/nas/kuro/KuroSiwo/"
path = os.path.join(
    folder_path, grid_dict["00f35c214e7b5cb493e0a9038ac74c0c"]['path'])
# %%

# 이벤트별로 VV와 VH 파일명을 매핑합니다.
# 여기서 'MS1'은 post-event, 'SL1'은 pre-event1, 'SL2'는 pre-event2를 의미합니다.
pairs = {
    'post_event': ('MS1_IVV_118_01_20150203.tif', 'MS1_IVH_118_01_20150203.tif'),
    'pre_event1': ('SL1_IVV_118_01_20141111.tif', 'SL1_IVH_118_01_20141111.tif'),
    'pre_event2': ('SL2_IVV_118_01_20141030.tif', 'SL2_IVH_118_01_20141030.tif')
}

# 각 이벤트에 대해 VV, VH 이미지를 읽어서 플롯합니다.
for event, (vv_file, vh_file) in pairs.items():
    # 전체 파일 경로 생성
    vv_path = os.path.join(path, vv_file)
    vh_path = os.path.join(path, vh_file)

    # rasterio로 각 tiff 파일 읽기
    with rasterio.open(vv_path) as vv_src:
        vv_data = vv_src.read(1)  # 첫 번째 밴드 읽기
    with rasterio.open(vh_path) as vh_src:
        vh_data = vh_src.read(1)

    # 이미지 플롯 (좌측: VV, 우측: VH)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(vv_data, cmap='gray')
    axes[0].set_title(f"{event} - VV")
    axes[0].axis('off')

    axes[1].imshow(vh_data, cmap='gray')
    axes[1].set_title(f"{event} - VH")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
