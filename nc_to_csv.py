import os
import pandas as pd
import numpy as np
import json
from netCDF4 import Dataset
from astropy.time import Time
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings(action='ignore')

# 실행 경로 설정
script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(script_path)
base_path = os.path.dirname(script_directory)

nc_data_path = os.path.join(base_path, "ENSEMBLE_MODEL_CROP_DATA")
save_path = os.path.join(base_path, "ENSEMBLE_MODEL_STATION_DATA")
os.makedirs(save_path, exist_ok=True)


# JSON 파일에서 target_lat, target_lon, target_date 불러오기
json_path = os.path.join(base_path, "target_metadata.json")
if not os.path.exists(json_path):
    raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {json_path}")

with open(json_path, 'r', encoding='utf-8') as f:
    target_metadata = json.load(f)

target_lat = target_metadata['target_lat']
target_lon = target_metadata['target_lon']
target_date = target_metadata['target_date']

print(f"JSON에서 target 정보 로드됨: 위도 {target_lat}, 경도 {target_lon}, 날짜 {target_date}")


# 가장 가까운 관측소 찾기 (SAR_meta_data.csv 활용)
meta_data_path = os.path.join(base_path, "SAR_meta_data.csv")
if not os.path.exists(meta_data_path):
    raise FileNotFoundError(f"관측소 메타데이터 파일을 찾을 수 없습니다: {meta_data_path}")

meta_df = pd.read_csv(meta_data_path, encoding='cp949')

# 하버사인 거리 계산 함수
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # 지구 반지름 (km)
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# target_lat, target_lon과 모든 관측소의 거리 계산
meta_df['distance'] = meta_df.apply(lambda row: haversine(target_lat, target_lon, row['lat'], row['lon']), axis=1)

# 가장 가까운 관측소 선택
nearest_station = meta_df.loc[meta_df['distance'].idxmin()]
station_num = nearest_station['stn_num']
station_name = nearest_station['stn_na']

print(f"가장 가까운 관측소: {station_num} ({station_name})")

# target_metadata.json에 가장 가까운 관측소 번호 추가
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(target_metadata, f, indent=4)

target_metadata['station_num'] = station_num
target_metadata['station_name'] = station_name
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(target_metadata, f, indent=4)

print(f"target_metadata.json에 station 정보 추가됨: {station_name} ({station_num})")

# target_date를 datetime 객체로 변환
target_datetime = datetime.strptime(target_date, "%Y%m%d%H")

# 10일치 날짜 리스트 생성 (12시간 간격 포함)
date_list = [(target_datetime - timedelta(hours=12 * i)).strftime("%Y%m%d%H") for i in range(21)]  # 10일치 12시간 간격

print(f"10일치(12시간 간격) nc 파일을 처리할 날짜 목록: {date_list}")



# 각 날짜별 NetCDF 파일을 처리하여 CSV로 저장
for date in date_list:
    nc_file = os.path.join(nc_data_path, f"cropped_{date}.nc")
    if not os.path.exists(nc_file):
        print(f"⚠️ 파일이 존재하지 않습니다: {nc_file}. 건너뜁니다.")
        continue

    print(f"📂 {nc_file} 처리 중...")
    nc = Dataset(nc_file)

    # NetCDF 파일에서 위도, 경도, 시간 데이터 로드
    lats = nc.variables['latitude'][:]
    lons = nc.variables['longitude'][:]
    time_data = nc.variables['time'][:]

    valid_points = 0

    # 300km 반경 내 모든 위도-경도 조합별로 처리
    for lat in lats:
        for lon in lons:
            if haversine(target_lat, target_lon, lat, lon) <= 300:  
                valid_points += 1
                lat_idx = np.abs(lats - lat).argmin()
                lon_idx = np.abs(lons - lon).argmin()
                print(f"  📍 저장할 지점: 위도 {lat:.2f}, 경도 {lon:.2f} (Index: LAT {lat_idx}, LON {lon_idx})")

                # 시간 데이터를 datetime으로 변환
                t_vec = Time(time_data, format='unix', scale='utc')
                t_vec_2 = t_vec.datetime  # datetime 리스트

                # DataFrame 생성 
                df = pd.DataFrame({'TIME': t_vec_2})

                for var_name, var_data in nc.variables.items():
                    if var_name not in ['latitude', 'longitude', 'time']:
                        if 'time' in var_data.dimensions:
                            df[var_name] = var_data[:, lat_idx, lon_idx]

                # 추가 정보 기입
                df['STN_NO'] = station_num
                df['LAT'] = lat
                df['LON'] = lon        
                df['LAT_I'] = lat_idx
                df['LON_J'] = lon_idx

                # CSV 저장 (파일명: {station_name}_{station_num}_{lat}_{lon}_{date}.csv)
                csv_file = os.path.join(save_path, f"{station_num}_{lat:.2f}_{lon:.2f}_{date}.csv")
                df.to_csv(csv_file, encoding='utf-8-sig', index=False)
                print(f"저장 완료: {csv_file}")

    nc.close()
print(f"300km 반경 내 (lat, lon) 조합 개수: {valid_points}")
print("10일치(12시간 간격) NetCDF → CSV 변환 완료")