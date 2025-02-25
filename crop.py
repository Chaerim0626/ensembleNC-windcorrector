import os
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from netCDF4 import Dataset
import warnings
import subprocess
import sys

warnings.filterwarnings(action='ignore')

# 사고지점 입력
target_lat = float(input("📍 분석할 위도를 입력하세요 (예: 35.52): "))
target_lon = float(input("📍 분석할 경도를 입력하세요 (예: 130.06): "))
target_date = input("📆 분석할 날짜를 입력하세요 (예: 2025020300): ")

# 실행 경로 설정
script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(script_path)
base_path = os.path.dirname(script_directory)

# NetCDF 데이터 경로 설정
nc_data_path = os.path.join(base_path, "ENSEMBLE_MODEL_DATA")
save_path = os.path.join(base_path, "ENSEMBLE_MODEL_CROP_DATA")
os.makedirs(save_path, exist_ok=True)


# 선택한 target 값 JSON 파일로 저장 
json_path = os.path.join(base_path, "target_metadata.json")
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump({'target_lat': target_lat, 'target_lon': target_lon, 'target_date': target_date}, f)

print(f"입력된 좌표 및 날짜 정보가 {json_path}에 저장되었습니다.")

# 10일 전까지의 날짜 리스트 생성 (12시간 간격)
target_datetime = datetime.strptime(target_date[:8] + "00", "%Y%m%d%H")  # 00시로 강제 변경

# 시작 시간: target_date 10일 전 (00시)
start_time = target_datetime - timedelta(days=10)

# 12시간 간격으로 생성 (종료 시간 포함)
date_list = []
current_time = start_time
while current_time <= target_datetime:
    date_list.append(current_time.strftime("%Y%m%d%H"))
    current_time += timedelta(hours=12)

print(date_list)


# 크롭할 영역 설정 (최대 300km 반경)
max_distance_km = 300
km_per_degree_lat = 111.32  
km_per_degree_lon = lambda lat: 111.32 * np.cos(np.radians(lat)) 

half_distance_km = max_distance_km / 2
lat_range = half_distance_km / km_per_degree_lat
lon_range = half_distance_km / km_per_degree_lon(target_lat)

lat_min, lat_max = target_lat - lat_range, target_lat + lat_range
lon_min, lon_max = target_lon - lon_range, target_lon + lon_range

# 날짜별로 NetCDF 파일 크롭
for date in date_list:
    nc_file = os.path.join(nc_data_path, f"ensemble.BMA_3d.{date}.wind.nc")
    
    if not os.path.exists(nc_file):
        print(f"⚠️ {nc_file} 파일이 없습니다. 건너뜁니다.")
        continue

    print(f"📂 NetCDF 파일 로드 중: {nc_file}")
    nc = Dataset(nc_file)

    # NetCDF 파일에서 위도/경도 데이터 확인
    latitudes = nc.variables["latitude"][:]  
    longitudes = nc.variables["longitude"][:] 

    lat_idx_min = np.abs(latitudes - lat_min).argmin()
    lat_idx_max = np.abs(latitudes - lat_max).argmin()
    lon_idx_min = np.abs(longitudes - lon_min).argmin()
    lon_idx_max = np.abs(longitudes - lon_max).argmin()

    # 데이터 크롭
    cropped_data = {}
    for var_name, var_data in nc.variables.items():
        if var_name not in ['latitude', 'longitude', 'time']: 
            if 'time' in var_data.dimensions:
                cropped_data[var_name] = var_data[:, lat_idx_min:lat_idx_max, lon_idx_min:lon_idx_max]
            else:  
                cropped_data[var_name] = var_data[lat_idx_min:lat_idx_max, lon_idx_min:lon_idx_max]

    # 새로운 NetCDF 파일 저장
    new_nc_path = os.path.join(save_path, f"cropped_{date}.nc")

    with Dataset(new_nc_path, 'w', format='NETCDF4') as new_nc:
        new_nc.createDimension('latitude', len(latitudes[lat_idx_min:lat_idx_max]))
        new_nc.createDimension('longitude', len(longitudes[lon_idx_min:lon_idx_max]))
        new_nc.createDimension('time', len(nc.variables['time']))

        new_latitudes = new_nc.createVariable('latitude', 'f4', ('latitude',))
        new_longitudes = new_nc.createVariable('longitude', 'f4', ('longitude',))
        new_times = new_nc.createVariable('time', 'f4', ('time',))

        new_latitudes[:] = latitudes[lat_idx_min:lat_idx_max]
        new_longitudes[:] = longitudes[lon_idx_min:lon_idx_max]
        new_times[:] = nc.variables['time'][:]

        for var_name, data in cropped_data.items():
            new_var = new_nc.createVariable(var_name, 'f4', ('time', 'latitude', 'longitude'))
            new_var[:] = data

    print(f"크롭된 데이터가 저장되었습니다: {new_nc_path}")


print("10일치 데이터 크롭 완료.")
