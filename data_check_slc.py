# %%
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
# %%
# Read the image file
folder_path = '15/slc_images'
# folder_path = '/home/junghwan/nas/kuro/KuroSiwo/118/01/0600fb567de25ba4b8055434e28f665c'
# folder_path = '/home/junghwan/nas/kuro/KuroSiwo/174/01/92d548f9c619569c92db790315c2eeb4'
# folder_path = '/home/junghwan/nas/kuro/SLC/174/01/7_14'
# folder_path = '/home/junghwan/nas/kuro/SLC/1111002/01/4_4'
# folder_path = '/home/junghwan/nas/kuro/KuroSiwo/1111002/01/992e25e133c259c18c5f2583dc9d8ce6'
file_paths = glob(folder_path + '/*.tif')

file_path = file_paths[0]
# file_path = folder_path + '/MK0_MLU_174_20150720_7_14_valid.tif'
with rasterio.open(file_path) as src:
    # Read all bands

    bands = [src.read(i) for i in range(1, src.count + 1)]
    print(src.__dict__)
    # Plot all bands
    if len(bands) == 1:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        band = bands[0]
        # Normalize the band
        # TODO plot 제대로
        if np.max(band) > 100:
            band = np.log10(band)
        band_min, band_max = band.min(), band.max()
        band_normalized = (band - band_min) / (band_max - band_min)

        # Plot the normalized band
        ax.imshow(band_normalized)
        ax.set_title('Band 1')
        ax.axis('off')
    else:
        fig, axes = plt.subplots(1, len(bands), figsize=(15, 5))
        for i, band in enumerate(bands):
            ax = axes[i]
            # Normalize the band
            if np.max(band) > 100:
                band = np.log10(band)
            band_min, band_max = band.min(), band.max()
            band_normalized = (band - band_min) / (band_max - band_min)

            # Plot the normalized band
            ax.imshow(band_normalized, cmap='gray')
            ax.set_title(f'Band {i+1}')
            ax.axis('off')
    plt.suptitle(file_path.split('/')[-1])
plt.show()

# %%
# Read the shapefile
shapefile_path = '15/event/event.dbf'
gdf = gpd.read_file(shapefile_path)

# Print the GeoDataFrame
print(gdf)

# Plot the shapefile
gdf.plot()
plt.title('Shapefile Plot')
plt.show()
# %%


pre_sar_image_path1 = '15/slc_images/S1A_IW_SLC__1SDV_20210908T191537_20210908T191604_039594_04AE27_B611.tif'
pre_sar_image_path2 = '15/slc_images/S1A_IW_SLC__1SDV_20210920T191538_20210920T191605_039769_04B418_5FF5.tif'
post_sar_image_path = '15/slc_images/S1A_IW_SLC__1SDV_20220307T191534_20220307T191601_042219_050811_2EE3.tif'

sar_image_path = pre_sar_image_path2
aoi_shp_path = '15/aoi/aoi.shp'
eventfile_path = '15/event/event.shp'
hydrofile_path = '15/hydro/hydroA.shp'

# pre_sar_image_path1 = '13/slc_images/S1A_IW_SLC__1SDV_20210807T201702_20210807T201729_039128_049E2A_20E9.tif'
# pre_sar_image_path2 = '13/slc_images/S1A_IW_SLC__1SDV_20210819T201703_20210819T201730_039303_04A432_B3DB.tif'
# post_sar_image_path = '13/slc_images/S1A_IW_SLC__1SDV_20220203T201701_20220203T201728_041753_04F7FA_08C4.tif'

# sar_image_path = post_sar_image_path
# aoi_shp_path = '13/aoi/aoi.shp'
# eventfile_path = '13/event/event.shp'
# hydrofile_path = '13/hydro/hydroA.shp'


# AOI와 event 데이터 읽기
aoi = gpd.read_file(aoi_shp_path)
event = gpd.read_file(eventfile_path)
hydro = gpd.read_file(hydrofile_path)

# SAR 이미지의 CRS 확인 및 AOI, event 재투영 (모두 EPSG:3857)
with rasterio.open(sar_image_path) as src:
    raster_crs = src.crs
print("SAR CRS:", raster_crs)
aoi = aoi.to_crs(raster_crs)
event = event.to_crs(raster_crs)
hydro = hydro.to_crs(raster_crs)

# SAR 이미지 클리핑 (AOI 영역)
with rasterio.open(sar_image_path) as src:
    clipped_image, clipped_transform = rasterio.mask.mask(
        src, aoi.geometry, crop=True)

# 올바른 순서로 클리핑된 이미지의 좌표(extent) 계산
# array_bounds는 (minx, miny, maxx, maxy)를 반환함.
min_x_img, min_y_img, max_x_img, max_y_img = array_bounds(
    clipped_image.shape[1], clipped_image.shape[2], clipped_transform)
# imshow의 extent는 [left, right, bottom, top]
img_extent = [min_x_img, max_x_img, min_y_img, max_y_img]

# AOI의 total_bounds: [minx, miny, maxx, maxy]
aoi_bounds = aoi.total_bounds

# 두 extents의 union 계산
min_x_union = min(img_extent[0], aoi_bounds[0])
max_x_union = max(img_extent[1], aoi_bounds[2])
min_y_union = min(img_extent[2], aoi_bounds[1])
max_y_union = max(img_extent[3], aoi_bounds[3])

# 플롯 생성
fig, ax = plt.subplots(figsize=(10, 10))

# SAR 이미지 (예: 두 번째 밴드를 로그 변환) 플롯 (낮은 zorder)
# 필요한 경우 밴드 번호를 조정하세요.
image = np.log10(clipped_image[2]+1)
img = ax.imshow(image, cmap='gray', extent=img_extent, zorder=1)
plt.colorbar(img, ax=ax, fraction=0.036, pad=0.04)

# AOI와 event 플롯 (높은 zorder)
# aoi.plot(ax=ax, facecolor='red', edgecolor='none', alpha=0.3, zorder=2)
aoi.boundary.plot(ax=ax, edgecolor='blue', linewidth=2, zorder=3)
event.plot(ax=ax, facecolor='none', edgecolor='green',
           linewidth=2, alpha=0.7, zorder=4)
hydro.plot(ax=ax, facecolor='none', edgecolor='yellow',
           linewidth=2, alpha=0.7, zorder=4)

# 두 데이터의 union 범위로 축을 설정
ax.set_xlim(min_x_union, max_x_union)
ax.set_ylim(min_y_union, max_y_union)

ax.set_title("Clipped SAR Image with AOI and Event Overlay")
ax.set_xlabel("Easting")
ax.set_ylabel("Northing")

plt.show()
# %%
# 각 영역의 geometry를 합치기 (union)
# AOI는 전체 영역, event와 hydro는 각각의 영역
aoi_union = aoi.unary_union
event_union = event.unary_union
hydro_union = hydro.unary_union

# event와 hydro의 영역을 합침 (union)
combined_union = event_union.union(hydro_union)

# AOI에서 event와 hydro 영역을 제외 (difference)
aoi_diff = aoi_union.difference(combined_union)

# aoi_diff는 Polygon 또는 MultiPolygon일 수 있으므로 리스트 형태로 변환
if aoi_diff.geom_type == 'Polygon':
    mask_geom = [aoi_diff]
elif aoi_diff.geom_type == 'MultiPolygon':
    mask_geom = list(aoi_diff.geoms)
else:
    mask_geom = [aoi_diff]


def get_area_pixels(sar_image_path, event, hydro, mask_geom):
    with rasterio.open(sar_image_path) as src:
        # SAR 이미지에서 AOI의 나머지 영역(즉, event와 hydro를 제외한 영역) 픽셀 값 추출
        normal_pixels, normal_transform = rasterio.mask.mask(
            src, mask_geom, crop=True)
        # event 영역에 해당하는 픽셀 값만 추출
        event_pixels, event_transform = rasterio.mask.mask(
            src, event.geometry, crop=True)
        # 물이 있는 영역에 해당하는 픽셀 값만 추출
        hydro_pixels, hydro_transform = rasterio.mask.mask(
            src, hydro.geometry, crop=True)
        clipped_image, clipped_transform = rasterio.mask.mask(
            src, aoi.geometry, crop=True)

    return clipped_image, normal_pixels, event_pixels, hydro_pixels


pre_clipped_image1, pre_normal_pixels1, pre_event_pixels1, pre_hydro_pixels1 = get_area_pixels(
    pre_sar_image_path1, event, hydro, mask_geom)
pre_clipped_image2, pre_normal_pixels2, pre_event_pixels2, pre_hydro_pixels2 = get_area_pixels(
    pre_sar_image_path2, event, hydro, mask_geom)
post_clipped_image, post_normal_pixels, post_event_pixels, post_hydro_pixels = get_area_pixels(
    post_sar_image_path, event, hydro, mask_geom)


def match_shapes(arr1, arr2, arr3):
    min_height = min(arr1.shape[0], arr2.shape[0], arr3.shape[0])
    min_width = min(arr1.shape[1], arr2.shape[1], arr3.shape[1])
    arr1, arr2, arr3 = arr1[:min_height, :min_width], arr2[:min_height,
                                                           :min_width], arr3[:min_height, :min_width]
    idx = (arr2 != 0) & (arr3 != 0) & (arr1 != 0)
    return arr1[idx], arr2[idx], arr3[idx]


def normalize(arr):
    # arr = (arr-np.mean(arr))/np.std(arr)
    arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    return arr


# Calculate differences for event
pre_event_pixels1_phase, pre_event_pixels2_phase, post_event_pixels_phase = match_shapes(
    pre_event_pixels1[3], pre_event_pixels2[3], post_event_pixels[3])
pre_pre_phase_diff_event = pre_event_pixels2_phase - pre_event_pixels1_phase
pre_post_phase_diff_event = post_event_pixels_phase - pre_event_pixels2_phase

pre_event_pixels1_amp, pre_event_pixels2_amp, post_event_pixels_amp = match_shapes(
    pre_event_pixels1[2], pre_event_pixels2[2], post_event_pixels[2])
pre_pre_amp_diff_event = normalize(np.log10(
    pre_event_pixels2_amp + 1)) - normalize(np.log10(pre_event_pixels1_amp + 1))
pre_post_amp_diff_event = normalize(np.log10(
    post_event_pixels_amp + 1)) - normalize(np.log10(pre_event_pixels2_amp + 1))

# Calculate differences for hydro
pre_hydro_pixels1_phase, pre_hydro_pixels2_phase, post_hydro_pixels_phase = match_shapes(
    pre_hydro_pixels1[3], pre_hydro_pixels2[3], post_hydro_pixels[3])
pre_pre_phase_diff_hydro = pre_hydro_pixels2_phase - pre_hydro_pixels1_phase
pre_post_phase_diff_hydro = post_hydro_pixels_phase - pre_hydro_pixels1_phase

pre_hydro_pixels1_amp, pre_hydro_pixels2_amp, post_hydro_pixels_amp = match_shapes(
    pre_hydro_pixels1[2], pre_hydro_pixels2[2], post_hydro_pixels[2])
pre_pre_amp_diff_hydro = normalize(np.log10(
    pre_hydro_pixels2_amp + 1)) - normalize(np.log10(pre_hydro_pixels1_amp + 1))
pre_post_amp_diff_hydro = normalize(np.log10(
    post_hydro_pixels_amp + 1)) - normalize(np.log10(pre_hydro_pixels1_amp + 1))

# Calculate differences for normal
pre_normal_pixels1_phase, pre_normal_pixels2_phase, post_normal_pixels_phase = match_shapes(
    pre_normal_pixels1[3], pre_normal_pixels2[3], post_normal_pixels[3])
pre_pre_phase_diff_normal = pre_normal_pixels2_phase - pre_normal_pixels1_phase
pre_post_phase_diff_normal = post_normal_pixels_phase - pre_normal_pixels2_phase

pre_normal_pixels1_amp, pre_normal_pixels2_amp, post_normal_pixels_amp = match_shapes(
    pre_normal_pixels1[2], pre_normal_pixels2[2], post_normal_pixels[2])
pre_pre_amp_diff_normal = normalize(np.log10(
    pre_normal_pixels2_amp + 1)) - normalize(np.log10(pre_normal_pixels1_amp + 1))
pre_post_amp_diff_normal = normalize(np.log10(
    post_normal_pixels_amp + 1)) - normalize(np.log10(pre_normal_pixels2_amp + 1))


def flatten_and_clean(diff_array):
    # Flatten and clean the arrays
    diff_flat = diff_array.flatten()
    return diff_flat[~np.isnan(diff_flat)]


pre_post_phase_diff_event_clean = flatten_and_clean(pre_post_phase_diff_event)
pre_post_amp_diff_event_clean = flatten_and_clean(pre_post_amp_diff_event)
pre_pre_amp_diff_event_clean = flatten_and_clean(pre_pre_amp_diff_event)
pre_pre_phase_diff_event_clean = flatten_and_clean(pre_pre_phase_diff_event)

pre_post_phase_diff_hydro_clean = flatten_and_clean(pre_post_phase_diff_hydro)
pre_post_amp_diff_hydro_clean = flatten_and_clean(pre_post_amp_diff_hydro)
pre_pre_phase_diff_hydro_clean = flatten_and_clean(pre_pre_phase_diff_hydro)
pre_pre_amp_diff_hydro_clean = flatten_and_clean(pre_pre_amp_diff_hydro)

pre_post_phase_diff_normal_clean = flatten_and_clean(
    pre_post_phase_diff_normal)
pre_post_amp_diff_normal_clean = flatten_and_clean(pre_post_amp_diff_normal)
pre_pre_phase_diff_normal_clean = flatten_and_clean(
    pre_pre_phase_diff_normal)
pre_pre_amp_diff_normal_clean = flatten_and_clean(pre_pre_amp_diff_normal)


def plot_histogram(diff_clean, title, ax):
    # Plot histograms
    ax.hist(diff_clean, bins=50, color='skyblue', edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel("Difference")
    ax.set_ylabel("Frequency")
    ax.grid(True)

# NOTE: 물과 홍수 영역의 분포가 비슷하다.


fig, axes = plt.subplots(2, 2, figsize=(15, 10))
plot_histogram(pre_post_phase_diff_event_clean,
               "Pre-Post Phase Difference (Event)", axes[0, 0])
plot_histogram(pre_post_amp_diff_event_clean,
               "Pre-Post Amplitude Difference (Event)", axes[0, 1])
plot_histogram(pre_pre_phase_diff_event_clean,
               "Pre-Pre Phase Difference (Event)", axes[1, 0])
plot_histogram(pre_pre_amp_diff_event_clean,
               "Pre-Pre Amplitude Difference (Event)", axes[1, 1])
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
plot_histogram(pre_post_phase_diff_hydro_clean,
               "Pre-Post Phase Difference (Hydro)", axes[0, 0])
plot_histogram(pre_post_amp_diff_hydro_clean,
               "Pre-Post Amplitude Difference (Hydro)", axes[0, 1])
plot_histogram(pre_pre_phase_diff_hydro_clean,
               "Pre-Pre Phase Difference (Hydro)", axes[1, 0])
plot_histogram(pre_pre_amp_diff_hydro_clean,
               "Pre-Pre Amplitude Difference (Hydro)", axes[1, 1])
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
plot_histogram(pre_post_phase_diff_normal_clean,
               "Pre-Post Phase Difference (Normal)", axes[0, 0])
plot_histogram(pre_post_amp_diff_normal_clean,
               "Pre-Post Amplitude Difference (Normal)", axes[0, 1])
plot_histogram(pre_pre_phase_diff_normal_clean,
               "Pre-Pre Phase Difference (Normal)", axes[1, 0])
plot_histogram(pre_pre_amp_diff_normal_clean,
               "Pre-Pre Amplitude Difference (Normal)", axes[1, 1])
plt.tight_layout()
plt.show()

# %%


def plot_individual_histogram(pixel_array, title, ax):
    # Flatten and clean the arrays
    pixel_flat = pixel_array.flatten()
    pixel_clean = pixel_flat[~np.isnan(pixel_flat)]

    # Plot histograms
    ax.hist(pixel_clean, bins=50, color='skyblue', edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.grid(True)


fig, axes = plt.subplots(3, 2, figsize=(15, 15))
plot_individual_histogram(
    pre_event_pixels1[3], "Pre Event Phase 1", axes[0, 0])
plot_individual_histogram(
    pre_event_pixels2[3], "Pre Event Phase 2", axes[0, 1])
plot_individual_histogram(post_event_pixels[3], "Post Event Phase", axes[1, 0])
plot_individual_histogram(
    pre_event_pixels1[2], "Pre Event Amplitude 1", axes[1, 1])
plot_individual_histogram(
    pre_event_pixels2[2], "Pre Event Amplitude 2", axes[2, 0])
plot_individual_histogram(
    post_event_pixels[2], "Post Event Amplitude", axes[2, 1])
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(3, 2, figsize=(15, 15))
plot_individual_histogram(
    pre_hydro_pixels1[3], "Pre Hydro Phase 1", axes[0, 0])
plot_individual_histogram(
    pre_hydro_pixels2[3], "Pre Hydro Phase 2", axes[0, 1])
plot_individual_histogram(post_hydro_pixels[3], "Post Hydro Phase", axes[1, 0])
plot_individual_histogram(
    pre_hydro_pixels1[2], "Pre Hydro Amplitude 1", axes[1, 1])
plot_individual_histogram(
    pre_hydro_pixels2[2], "Pre Hydro Amplitude 2", axes[2, 0])
plot_individual_histogram(
    post_hydro_pixels[2], "Post Hydro Amplitude", axes[2, 1])
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(3, 2, figsize=(15, 15))
plot_individual_histogram(
    pre_normal_pixels1[3], "Pre Normal Phase 1", axes[0, 0])
plot_individual_histogram(
    pre_normal_pixels2[3], "Pre Normal Phase 2", axes[0, 1])
plot_individual_histogram(
    post_normal_pixels[3], "Post Normal Phase", axes[1, 0])
plot_individual_histogram(
    pre_normal_pixels1[2], "Pre Normal Amplitude 1", axes[1, 1])
plot_individual_histogram(
    pre_normal_pixels2[2], "Pre Normal Amplitude 2", axes[2, 0])
plot_individual_histogram(
    post_normal_pixels[2], "Post Normal Amplitude", axes[2, 1])
plt.tight_layout()
plt.show()
