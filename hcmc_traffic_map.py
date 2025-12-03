import streamlit as st
import osmnx as ox
import folium
from streamlit_folium import st_folium
import pandas as pd
import warnings
import pickle
import os
import hashlib
import json
from datetime import datetime, timedelta
import numpy as np
import math
import gzip
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import time
from shapely.geometry import LineString, Point
import geopandas as gpd
from typing import List, Tuple, Dict, Any

# Ẩn cảnh báo
warnings.filterwarnings('ignore')

# Cấu hình trang web (title, layout)
st.set_page_config(
    page_title="Bản Đồ Giao Thông TP.HCM",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ẩn các phần tử mặc định của Streamlit (Menu, Footer)
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Cấu hình OSMnx
ox.settings.timeout = 1000
ox.settings.use_cache = True
ox.settings.log_console = False
ox.settings.max_query_area_size = 2500000000

# Tạo thư mục cache nếu chưa tồn tại
CACHE_DIR = "map_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Danh sách gợi ý sẵn
DISTRICTS = {
    "Quận 1": "District 1, Ho Chi Minh City, Vietnam",
    "Quận 3": "District 3, Ho Chi Minh City, Vietnam",
    "Quận 4": "District 4, Ho Chi Minh City, Vietnam",
    "Quận 5": "District 5, Ho Chi Minh City, Vietnam",
    "Quận 6": "District 6, Ho Chi Minh City, Vietnam",
    "Quận 7": "District 7, Ho Chi Minh City, Vietnam",
    "Quận 8": "District 8, Ho Chi Minh City, Vietnam",
    "Quận 10": "District 10, Ho Chi Minh City, Vietnam",
    "Quận 11": "District 11, Ho Chi Minh City, Vietnam",
    "Quận 12": "District 12, Ho Chi Minh City, Vietnam",
    "Quận Bình Thạnh": "Binh Thanh District, Ho Chi Minh City, Vietnam",
    "Quận Gò Vấp": "Go Vap District, Ho Chi Minh City, Vietnam",
    "Quận Phú Nhuận": "Phu Nhuan District, Ho Chi Minh City, Vietnam",
    "Quận Tân Bình": "Tan Binh District, Ho Chi Minh City, Vietnam",
    "Quận Tân Phú": "Tan Phu District, Ho Chi Minh City, Vietnam",
    "Quận Bình Tân": "Binh Tan District, Ho Chi Minh City, Vietnam",
    "TP. Thủ Đức": "Thu Duc City, Ho Chi Minh City, Vietnam",
    "Huyện Bình Chánh": "Binh Chanh District, Ho Chi Minh City, Vietnam",
    "Huyện Củ Chi": "Cu Chi District, Ho Chi Minh City, Vietnam",
    "Huyện Nhà Bè": "Nha Be District, Ho Chi Minh City, Vietnam",
    "Huyện Hóc Môn": "Hoc Mon District, Ho Chi Minh City, Vietnam",
    "Huyện Cần Giờ": "Can Gio District, Ho Chi Minh City, Vietnam",
    "Toàn Thành Phố (Rất Chậm 🐢)": "Ho Chi Minh City, Vietnam"
}

# Biến toàn cục để cache trong bộ nhớ
_MEMORY_CACHE = {}
_PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL

# Hằng số cho tính toán nhanh
_EARTH_RADIUS = 6371000
_DEG_TO_RAD = math.pi / 180.0
_RAD_TO_DEG = 180.0 / math.pi


@lru_cache(maxsize=5000)
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Tính khoảng cách Haversine với caching tối ưu"""
    if lat1 == lat2 and lon1 == lon2:
        return 0.0

    # Chuyển đổi độ sang radian
    lat1_rad = lat1 * _DEG_TO_RAD
    lon1_rad = lon1 * _DEG_TO_RAD
    lat2_rad = lat2 * _DEG_TO_RAD
    lon2_rad = lon2 * _DEG_TO_RAD

    # Chênh lệch tọa độ
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Công thức Haversine tối ưu hóa
    sin_dlat_2 = math.sin(dlat * 0.5)
    sin_dlon_2 = math.sin(dlon * 0.5)
    a = sin_dlat_2 * sin_dlat_2 + math.cos(lat1_rad) * math.cos(lat2_rad) * sin_dlon_2 * sin_dlon_2

    # Tránh giá trị ngoài khoảng [-1, 1] do sai số số học
    if a > 1.0:
        a = 1.0
    elif a < 0.0:
        a = 0.0

    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return _EARTH_RADIUS * c


def calculate_route_length_optimized(coords: List[Tuple[float, float]]) -> float:
    """Tính chiều dài tuyến đường với tối ưu hóa vectorized"""
    if len(coords) < 2:
        return 0.0

    # Chuyển đổi tọa độ sang numpy array để tính toán vectorized
    coords_array = np.array(coords)
    lats = coords_array[:, 0] * _DEG_TO_RAD
    lons = coords_array[:, 1] * _DEG_TO_RAD

    # Tính sin và cos cho tất cả các điểm
    cos_lats = np.cos(lats)
    sin_lats = np.sin(lats)

    # Tính khoảng cách giữa các điểm liên tiếp
    cos_dlat = np.cos(lats[1:] - lats[:-1])
    cos_dlon = np.cos(lons[1:] - lons[:-1])

    # Công thức haversine vectorized
    a = sin_lats[:-1] * sin_lats[1:] + cos_lats[:-1] * cos_lats[1:] * cos_dlon
    a = np.clip(a, -1.0, 1.0)  # Đảm bảo giá trị trong khoảng [-1, 1]

    distances = _EARTH_RADIUS * np.arccos(a)

    return float(np.sum(distances))


def preprocess_edges_for_fast_drawing(edges: gpd.GeoDataFrame) -> Dict[str, Any]:
    """Tiền xử lý edges để vẽ nhanh hơn"""
    processed_data = {
        'coords_list': [],
        'weights': [],
        'names': [],
        'highways': [],
        'lengths': [],
        'total_length': 0.0
    }

    total_length = 0.0
    max_edges = min(len(edges), 50000)  # Giới hạn số lượng đường vẽ

    for idx, row in edges.head(max_edges).iterrows():
        try:
            if hasattr(row.geometry, 'coords'):
                # Lấy tọa độ và chuyển đổi thành list
                coords = [(lat, lon) for lon, lat in row.geometry.coords]
                if len(coords) >= 2:
                    processed_data['coords_list'].append(coords)

                    # Tính chiều dài
                    length = calculate_route_length_optimized(coords)
                    total_length += length
                    processed_data['lengths'].append(length)

                    # Xác định độ dày dựa trên loại đường
                    hw = row.get('highway', 'unclassified')
                    if isinstance(hw, list):
                        hw = hw[0] if hw else 'unclassified'

                    # Phân loại đường và gán weight
                    if hw in ['motorway', 'trunk']:
                        weight = 4.0
                    elif hw == 'primary':
                        weight = 3.5
                    elif hw == 'secondary':
                        weight = 3.0
                    elif hw == 'tertiary':
                        weight = 2.5
                    elif hw in ['residential', 'living_street', 'unclassified']:
                        weight = 2.0
                    elif hw == 'service':
                        weight = 1.5
                    else:
                        weight = 2.0

                    processed_data['weights'].append(weight)
                    processed_data['names'].append(row.get('name', 'Đường không tên'))
                    processed_data['highways'].append(hw)

        except Exception:
            continue

    processed_data['total_length'] = total_length
    return processed_data


class FastMapRenderer:
    """Lớp render bản đồ nhanh với tối ưu hóa"""

    def __init__(self):
        self.color_palette = {
            'motorway': '#FF6B6B',  # Đỏ
            'trunk': '#FF8E53',  # Cam
            'primary': '#FFD166',  # Vàng
            'secondary': '#06D6A0',  # Xanh lá
            'tertiary': '#118AB2',  # Xanh dương
            'residential': '#9D4EDD',  # Tím
            'default': '#6C757D'  # Xám
        }

        self.weight_config = {
            'motorway': 4.5,
            'trunk': 4.0,
            'primary': 3.5,
            'secondary': 3.0,
            'tertiary': 2.5,
            'residential': 2.0,
            'service': 1.5,
            'default': 2.0
        }

    def get_color_for_highway(self, highway_type: str) -> str:
        """Lấy màu dựa trên loại đường"""
        for key in self.color_palette:
            if key in str(highway_type):
                return self.color_palette[key]
        return self.color_palette['default']

    def get_weight_for_highway(self, highway_type: str) -> float:
        """Lấy độ dày dựa trên loại đường"""
        for key in self.weight_config:
            if key in str(highway_type):
                return self.weight_config[key]
        return self.weight_config['default']

    def render_edges_batch(self, m: folium.Map, processed_data: Dict[str, Any],
                           max_edges_per_batch: int = 1000) -> int:
        """Render edges theo batch để tăng tốc độ"""
        coords_list = processed_data['coords_list']
        weights = processed_data['weights']
        names = processed_data['names']
        highways = processed_data['highways']
        lengths = processed_data['lengths']

        total_edges = len(coords_list)
        rendered_count = 0

        # Tạo FeatureGroup cho từng loại đường để tối ưu hóa
        feature_groups = {}

        # Nhóm các đường theo loại để render cùng lúc
        for i in range(min(total_edges, max_edges_per_batch)):
            try:
                coords = coords_list[i]
                weight = weights[i]
                name = names[i]
                highway = highways[i]
                length = lengths[i]

                # Tạo popup thông tin
                popup_text = f"""
                <div style="font-family: Arial; font-size: 12px; min-width: 200px;">
                    <b>🏛️ Tên đường:</b> {name}<br>
                    <b>🛣️ Loại đường:</b> {highway}<br>
                    <b>📏 Chiều dài:</b> {length:.0f}m<br>
                    <b>🎨 Màu sắc:</b> Theo loại đường
                </div>
                """

                # Lấy màu dựa trên loại đường
                color = self.get_color_for_highway(highway)

                # Tạo PolyLine với popup
                polyline = folium.PolyLine(
                    locations=coords,
                    color=color,
                    weight=weight,
                    opacity=0.9,
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=f"{name} ({highway}) - {length:.0f}m"
                )

                # Thêm vào bản đồ
                polyline.add_to(m)
                rendered_count += 1

                # Hiển thị tiến trình
                if rendered_count % 500 == 0:
                    st.text(f"🖌️ Đã vẽ {rendered_count} đường...")

            except Exception as e:
                continue

        return rendered_count


class CacheManager:
    """Quản lý cache với tối ưu hóa nâng cao"""

    @staticmethod
    def get_cache_key(place_name: str, detailed: bool = False) -> str:
        """Tạo key cache với hashing hiệu quả"""
        cache_string = f"{place_name}_{detailed}_v2"
        return hashlib.sha256(cache_string.encode()).hexdigest()[:16]

    @staticmethod
    def get_cache_info_path() -> str:
        """Lấy đường dẫn file thông tin cache"""
        return os.path.join(CACHE_DIR, "cache_info_v2.json")

    @staticmethod
    def get_cache_file_path(cache_key: str, compressed: bool = True) -> str:
        """Lấy đường dẫn file cache"""
        if compressed:
            return os.path.join(CACHE_DIR, f"{cache_key}.pkl.gz")
        else:
            return os.path.join(CACHE_DIR, f"{cache_key}.pkl")

    @staticmethod
    def get_metadata_file_path(cache_key: str) -> str:
        """Lấy đường dẫn file metadata"""
        return os.path.join(CACHE_DIR, f"{cache_key}_meta.json")

    @staticmethod
    def load_cache_info() -> Dict:
        """Tải thông tin cache"""
        info_path = CacheManager.get_cache_info_path()
        if os.path.exists(info_path):
            try:
                with open(info_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    @staticmethod
    def save_cache_info(cache_info: Dict):
        """Lưu thông tin cache"""
        info_path = CacheManager.get_cache_info_path()
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(cache_info, f, ensure_ascii=False, indent=2)

    @staticmethod
    def is_cache_valid(cache_key: str, max_age_days: int = 30) -> bool:
        """Kiểm tra cache còn hợp lệ"""
        meta_path = CacheManager.get_metadata_file_path(cache_key)
        if not os.path.exists(meta_path):
            return False

        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            created_time = datetime.fromisoformat(metadata.get('created_at', '2000-01-01'))
            age = datetime.now() - created_time

            return age.days < max_age_days
        except:
            return False

    @staticmethod
    def update_cache_metadata(cache_key: str, place_name: str, edges_count: int,
                              total_length_km: float, detailed: bool = False,
                              compressed: bool = True):
        """Cập nhật metadata cho cache"""
        cache_file_path = CacheManager.get_cache_file_path(cache_key, compressed)
        file_size_kb = 0
        if os.path.exists(cache_file_path):
            file_size_kb = os.path.getsize(cache_file_path) / 1024

        metadata = {
            'place_name': place_name,
            'detailed': detailed,
            'edges_count': edges_count,
            'total_length_km': total_length_km,
            'created_at': datetime.now().isoformat(),
            'size_kb': file_size_kb,
            'compressed': compressed,
            'version': '2.0'
        }

        meta_path = CacheManager.get_metadata_file_path(cache_key)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        cache_info = CacheManager.load_cache_info()
        cache_info[cache_key] = metadata
        CacheManager.save_cache_info(cache_info)

    @staticmethod
    def save_cache_data(cache_key: str, data: Any, compressed: bool = True) -> bool:
        """Lưu dữ liệu cache với tối ưu hóa"""
        cache_file_path = CacheManager.get_cache_file_path(cache_key, compressed)

        try:
            if compressed:
                # Sử dụng gzip với mức nén tối ưu
                with gzip.open(cache_file_path, 'wb', compresslevel=6) as f:
                    pickle.dump(data, f, protocol=_PICKLE_PROTOCOL)
            else:
                with open(cache_file_path, 'wb') as f:
                    pickle.dump(data, f, protocol=_PICKLE_PROTOCOL)
            return True
        except Exception as e:
            st.warning(f"⚠️ Lỗi khi lưu cache: {e}")
            return False

    @staticmethod
    def load_cache_data(cache_key: str, compressed: bool = True) -> Any:
        """Tải dữ liệu cache"""
        cache_file_path = CacheManager.get_cache_file_path(cache_key, compressed)

        if not os.path.exists(cache_file_path):
            return None

        try:
            if compressed:
                with gzip.open(cache_file_path, 'rb') as f:
                    return pickle.load(f)
            else:
                with open(cache_file_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            st.warning(f"⚠️ Lỗi khi đọc cache: {e}")
            return None


def get_graph_data(place_name: str, detailed: bool = False) -> gpd.GeoDataFrame:
    """Lấy dữ liệu đồ thị với caching tối ưu"""

    cache_key = CacheManager.get_cache_key(place_name, detailed)
    compressed = True

    # 1. Kiểm tra cache trong bộ nhớ
    if cache_key in _MEMORY_CACHE:
        edges, metadata = _MEMORY_CACHE[cache_key]
        st.info(f"⚡ Đang tải từ bộ nhớ: {metadata['edges_count']} tuyến đường")
        return edges

    # 2. Kiểm tra cache trên đĩa
    if CacheManager.is_cache_valid(cache_key):
        try:
            with st.spinner("🚀 Đang đọc dữ liệu từ cache..."):
                edges = CacheManager.load_cache_data(cache_key, compressed)

                if edges is not None:
                    # Đọc metadata
                    meta_path = CacheManager.get_metadata_file_path(cache_key)
                    if os.path.exists(meta_path):
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                    else:
                        metadata = {
                            'edges_count': len(edges),
                            'total_length_km': 0
                        }

                    # Lưu vào cache bộ nhớ
                    _MEMORY_CACHE[cache_key] = (edges, metadata)

                    st.success(f"✅ Đã tải từ cache: {len(edges)} tuyến đường")
                    return edges
        except Exception as e:
            st.warning(f"⚠️ Lỗi cache: {e}")

    # 3. Tải mới từ OSM
    return download_and_cache_data(place_name, detailed, cache_key, compressed)


def download_and_cache_data(place_name: str, detailed: bool,
                            cache_key: str, compressed: bool = True) -> gpd.GeoDataFrame:
    """Tải dữ liệu từ OSM và lưu cache"""

    # Xác định custom_filter dựa trên loại khu vực
    if detailed:
        custom_filter = '["highway"~"motorway|trunk|primary|secondary|tertiary|residential|service|living_street|unclassified|pedestrian"]'
        st.info("🔍 Đang tải chi tiết: Lấy cả đường nhỏ...")
    elif "Ho Chi Minh City" in place_name and "District" not in place_name:
        custom_filter = '["highway"~"motorway|trunk|primary|secondary"]'
    else:
        custom_filter = '["highway"~"motorway|trunk|primary|secondary|tertiary"]'

    try:
        with st.spinner(f"🌐 Đang tải dữ liệu từ OpenStreetMap..."):
            progress_bar = st.progress(0)

            # Tải dữ liệu với tiến trình
            G = ox.graph_from_place(
                place_name,
                network_type='drive',
                simplify=True,
                custom_filter=custom_filter,
                retain_all=True
            )

            progress_bar.progress(50)

            nodes, edges = ox.graph_to_gdfs(G)

            progress_bar.progress(80)

            # Tính tổng chiều dài gần đúng (nhanh hơn)
            if 'length' in edges.columns:
                total_length_km = edges['length'].sum() / 1000
            else:
                total_length_km = len(edges) * 0.1  # Ước lượng

            progress_bar.progress(100)
            progress_bar.empty()

            # Lưu cache
            if CacheManager.save_cache_data(cache_key, edges, compressed):
                CacheManager.update_cache_metadata(
                    cache_key, place_name, len(edges),
                    total_length_km, detailed, compressed
                )

                metadata = {
                    'place_name': place_name,
                    'detailed': detailed,
                    'edges_count': len(edges),
                    'total_length_km': total_length_km,
                    'created_at': datetime.now().isoformat(),
                    'size_kb': os.path.getsize(CacheManager.get_cache_file_path(cache_key, compressed)) / 1024,
                    'compressed': compressed
                }
                _MEMORY_CACHE[cache_key] = (edges, metadata)

                st.success(f"💾 Đã lưu cache: {len(edges)} tuyến đường, {total_length_km:.1f} km")

            return edges

    except Exception as e:
        st.error(f"❌ Lỗi khi tải dữ liệu: {str(e)}")
        # Thử phương pháp backup
        try:
            st.info("🔄 Đang thử phương pháp backup...")
            G = ox.graph_from_place(place_name, network_type='drive')
            nodes, edges = ox.graph_to_gdfs(G)
            return edges
        except:
            st.error("❌ Không thể tải dữ liệu. Vui lòng thử khu vực khác.")
            return gpd.GeoDataFrame()


class HCMTrafficMap:
    def __init__(self):
        self.cache_info = CacheManager.load_cache_info()
        self.renderer = FastMapRenderer()
        self.edges_data = None

    def create_sidebar(self) -> Tuple[str, str, bool]:
        """Tạo sidebar với các tùy chọn"""
        st.sidebar.title("⚙️ Tùy Chọn Bản Đồ")

        # Hiển thị thông tin cache
        self.display_cache_info()

        # Quản lý cache
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🗃️ Quản lý Cache")

        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🗑️ Xóa tất cả cache", help="Xóa toàn bộ dữ liệu đã lưu"):
                self.clear_all_cache()

        with col2:
            if st.button("🔄 Làm mới cache", help="Xóa cache và tải lại dữ liệu"):
                self.refresh_current_cache()

        # Tùy chọn hiển thị
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎨 Tùy chọn hiển thị")

        self.color_mode = st.sidebar.selectbox(
            "Chế độ màu:",
            ["Theo loại đường", "Màu duy nhất"],
            index=0,
            help="Chọn cách hiển thị màu sắc cho các tuyến đường"
        )

        self.opacity = st.sidebar.slider(
            "Độ trong suốt:",
            min_value=0.1,
            max_value=1.0,
            value=0.8,
            step=0.1,
            help="Điều chỉnh độ trong suốt của đường"
        )

        # Chọn khu vực
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📍 Chọn khu vực")

        options = list(DISTRICTS.keys()) + ["🔍 Nhập địa điểm tùy chỉnh..."]
        selection = st.sidebar.selectbox("Chọn khu vực:", options, index=0)

        # Tùy chọn chi tiết
        detailed_mode = False
        if selection == "Quận 1":
            detailed_mode = st.sidebar.checkbox(
                "🔎 Chế độ chi tiết (hẻm, ngõ)",
                value=True,
                help="Hiển thị cả các đường nhỏ, hẻm, ngõ"
            )

        # Xử lý lựa chọn
        if selection == "🔍 Nhập địa điểm tùy chỉnh...":
            st.sidebar.markdown("---")
            custom_input = st.sidebar.text_input(
                "Nhập địa điểm (tiếng Anh):",
                "Ben Thanh Market, District 1"
            )

            display_name = custom_input
            place_query = custom_input

            if not any(x in custom_input.lower() for x in ['vietnam', 'hcmc', 'ho chi minh']):
                place_query = custom_input + ", Ho Chi Minh City, Vietnam"
                st.sidebar.caption("📍 Đã thêm ', Ho Chi Minh City, Vietnam'")

            return place_query, display_name, detailed_mode
        else:
            return DISTRICTS[selection], selection, detailed_mode

    def display_cache_info(self):
        """Hiển thị thông tin cache"""
        if not self.cache_info:
            st.sidebar.info("📭 Chưa có dữ liệu cache")
            return

        total_size = sum(info.get('size_kb', 0) for info in self.cache_info.values())
        total_edges = sum(info.get('edges_count', 0) for info in self.cache_info.values())

        st.sidebar.markdown(f"### 📊 Thống kê Cache")
        st.sidebar.metric("Số khu vực", len(self.cache_info))
        st.sidebar.metric("Tổng tuyến đường", f"{total_edges:,}")
        st.sidebar.metric("Dung lượng", f"{total_size:.1f} KB")

        # Hiển thị danh sách cache
        st.sidebar.markdown("**Danh sách cache:**")
        for cache_key, info in list(self.cache_info.items())[:5]:
            name = info.get('place_name', 'Unknown')[:25] + ("..." if len(info.get('place_name', '')) > 25 else "")
            edges = info.get('edges_count', 0)
            st.sidebar.caption(f"• {name}: {edges} đường")

    def clear_all_cache(self):
        """Xóa tất cả cache"""
        try:
            _MEMORY_CACHE.clear()

            cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith(('.pkl', '.json', '.gz'))]
            deleted = 0

            for file in cache_files:
                try:
                    os.remove(os.path.join(CACHE_DIR, file))
                    deleted += 1
                except:
                    pass

            CacheManager.save_cache_info({})
            st.sidebar.success(f"✅ Đã xóa {deleted} file cache")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"❌ Lỗi: {e}")

    def refresh_current_cache(self):
        """Làm mới cache hiện tại"""
        try:
            # Xóa cache bộ nhớ
            _MEMORY_CACHE.clear()

            # Xóa file cache
            cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith(('.pkl', '.json', '.gz'))]
            for file in cache_files:
                try:
                    os.remove(os.path.join(CACHE_DIR, file))
                except:
                    pass

            CacheManager.save_cache_info({})
            st.sidebar.success("✅ Đã làm mới cache. Vui lòng tải lại trang.")
            time.sleep(2)
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"❌ Lỗi: {e}")

    def load_data(self, place_query: str, display_name: str, detailed: bool = False) -> gpd.GeoDataFrame:
        """Tải dữ liệu"""
        try:
            with st.spinner(f"📥 Đang tải: {display_name}..."):
                edges = get_graph_data(place_query, detailed)

                if edges is not None and not edges.empty:
                    # Tiền xử lý dữ liệu để vẽ nhanh
                    self.edges_data = preprocess_edges_for_fast_drawing(edges)

                    # Thống kê loại đường
                    if 'highway' in edges.columns:
                        highway_stats = edges['highway'].value_counts().head(10)

                        st.sidebar.markdown("---")
                        st.sidebar.markdown("### 🛣️ Thống kê loại đường")

                        for hw_type, count in highway_stats.items():
                            if isinstance(hw_type, list):
                                hw_type = hw_type[0] if hw_type else "unknown"
                            st.sidebar.caption(f"• {hw_type}: {count}")

                    st.success(f"✅ Đã tải: {display_name}")
                    st.info(f"📏 {len(edges)} tuyến đường, {self.edges_data['total_length'] / 1000:.1f} km")

                    return edges
                else:
                    st.error("❌ Không có dữ liệu đường cho khu vực này")
                    return None

        except Exception as e:
            st.error(f"❌ Lỗi khi tải: {str(e)[:100]}")
            return None

    def create_map(self, edges: gpd.GeoDataFrame) -> folium.Map:
        """Tạo bản đồ với tốc độ vẽ tối ưu"""
        # Tính tâm bản đồ
        if not edges.empty:
            bounds = edges.total_bounds
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lon = (bounds[0] + bounds[2]) / 2
            zoom_start = 14 if len(edges) > 1000 else 15
        else:
            center_lat, center_lon = 10.7769, 106.7009
            zoom_start = 14

        # Tạo bản đồ
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            tiles='OpenStreetMap',
            control_scale=True,
            prefer_canvas=True  # Sử dụng canvas để render nhanh hơn
        )

        # Thêm tile layer options
        folium.TileLayer(
            'CartoDB positron',
            name='Light Mode',
            attr='CartoDB'
        ).add_to(m)

        folium.TileLayer(
            'CartoDB dark_matter',
            name='Dark Mode',
            attr='CartoDB'
        ).add_to(m)

        folium.LayerControl().add_to(m)

        # Vẽ các đường
        if self.edges_data and len(self.edges_data['coords_list']) > 0:
            with st.spinner("🎨 Đang vẽ bản đồ..."):
                progress_bar = st.progress(0)

                # Render với batch processing
                rendered = self.renderer.render_edges_batch(
                    m,
                    self.edges_data,
                    max_edges_per_batch=min(2000, len(self.edges_data['coords_list']))
                )

                progress_bar.progress(100)
                progress_bar.empty()

                st.info(f"🖌️ Đã vẽ {rendered} tuyến đường")

        # Thêm marker cho các địa điểm quan trọng
        if "District 1" in str(edges.crs) if edges.crs else False:
            landmarks = [
                ("🏪 Chợ Bến Thành", 10.772, 106.698, "green"),
                ("🎭 Nhà hát Thành phố", 10.777, 106.703, "red"),
                ("🏛️ Bưu điện Trung tâm", 10.780, 106.699, "blue"),
                ("🏛️ Dinh Độc Lập", 10.777, 106.695, "purple"),
                ("⛵ Bến Bạch Đằng", 10.773, 106.706, "orange")
            ]

            for name, lat, lon, color in landmarks:
                folium.Marker(
                    location=[lat, lon],
                    popup=name,
                    icon=folium.Icon(color=color, icon='info-sign', prefix='fa')
                ).add_to(m)

        # Thêm fullscreen button
        folium.plugins.Fullscreen(
            position='topright',
            title='Xem toàn màn hình',
            title_cancel='Thoát toàn màn hình',
            force_separate_button=True
        ).add_to(m)

        # Thêm measure control
        folium.plugins.MeasureControl(
            position='topright',
            primary_length_unit='meters',
            secondary_length_unit='kilometers'
        ).add_to(m)

        return m


def main():
    """Hàm chính của ứng dụng"""
    # Header với styling
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: white; margin: 0;">🗺️ BẢN ĐỒ GIAO THÔNG TP.HCM</h1>
        <p style="color: #f0f0f0; margin: 5px 0 0 0;">Visualization & Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # Thông tin phiên bản
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 🚀 Phiên bản Tối ưu
    **Tính năng nổi bật:**
    • ⚡ Vẽ đường siêu tốc
    • 🎨 Màu sắc theo loại đường
    • 💾 Cache thông minh
    • 📊 Thống kê chi tiết
    """)

    # Khởi tạo ứng dụng
    app = HCMTrafficMap()

    # Tải dữ liệu
    place_query, display_name, detailed_mode = app.create_sidebar()

    if place_query:
        # Thông tin khu vực
        st.markdown(f"### 📍 Khu vực: **{display_name}**")
        if detailed_mode:
            st.info("🔍 **Chế độ chi tiết:** Hiển thị cả đường nhỏ, hẻm, ngõ")

        # Tải và hiển thị dữ liệu
        edges = app.load_data(place_query, display_name, detailed_mode)

        if edges is not None and not edges.empty:
            # Tạo và hiển thị bản đồ
            with st.spinner("🔄 Đang tạo bản đồ..."):
                traffic_map = app.create_map(edges)

                # Hiển thị bản đồ với kích thước lớn
                st_folium(
                    traffic_map,
                    width=1400,
                    height=700,
                    returned_objects=[]
                )

            # Hiển thị thông tin chi tiết
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Tổng tuyến đường", f"{len(edges):,}")

            with col2:
                total_km = app.edges_data['total_length'] / 1000 if app.edges_data else 0
                st.metric("Tổng chiều dài", f"{total_km:.1f} km")

            with col3:
                displayed = len(app.edges_data['coords_list']) if app.edges_data else 0
                st.metric("Đã hiển thị", f"{displayed:,}")

            with col4:
                if detailed_mode:
                    st.metric("Chế độ", "Chi tiết 🎯")
                else:
                    st.metric("Chế độ", "Thông thường ⚡")

            # Chú thích màu sắc
            st.markdown("### 🎨 Chú thích màu sắc đường")
            colors = app.renderer.color_palette
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(
                    f"<div style='background-color:{colors['motorway']}; padding: 10px; border-radius: 5px; color: white;'>🛣️ Motorway</div>",
                    unsafe_allow_html=True)
                st.markdown(
                    f"<div style='background-color:{colors['trunk']}; padding: 10px; border-radius: 5px; color: white; margin-top: 5px;'>🛣️ Trunk Road</div>",
                    unsafe_allow_html=True)

            with col2:
                st.markdown(
                    f"<div style='background-color:{colors['primary']}; padding: 10px; border-radius: 5px; color: black;'>🛣️ Primary</div>",
                    unsafe_allow_html=True)
                st.markdown(
                    f"<div style='background-color:{colors['secondary']}; padding: 10px; border-radius: 5px; color: white; margin-top: 5px;'>🛣️ Secondary</div>",
                    unsafe_allow_html=True)

            with col3:
                st.markdown(
                    f"<div style='background-color:{colors['tertiary']}; padding: 10px; border-radius: 5px; color: white;'>🛣️ Tertiary</div>",
                    unsafe_allow_html=True)
                st.markdown(
                    f"<div style='background-color:{colors['residential']}; padding: 10px; border-radius: 5px; color: white; margin-top: 5px;'>🏘️ Residential</div>",
                    unsafe_allow_html=True)

            with col4:
                st.markdown(
                    f"<div style='background-color:{colors['default']}; padding: 10px; border-radius: 5px; color: white;'>🛣️ Khác</div>",
                    unsafe_allow_html=True)
                st.markdown(
                    "<div style='padding: 10px; border-radius: 5px; background: #f0f0f0; margin-top: 5px;'>📏 Độ dày: Theo loại đường</div>",
                    unsafe_allow_html=True)

            # Hướng dẫn sử dụng
            with st.expander("📖 Hướng dẫn sử dụng nhanh"):
                st.markdown("""
                1. **Chọn khu vực** từ menu bên trái
                2. **Chế độ chi tiết** cho Quận 1: Hiển thị cả hẻm, ngõ
                3. **Tương tác với bản đồ:**
                   - Click vào đường để xem thông tin
                   - Zoom in/out bằng scroll chuột
                   - Kéo để di chuyển bản đồ
                   - Sử dụng nút toàn màn hình góc trên phải
                4. **Màu sắc đường:** Mỗi loại đường có màu riêng
                5. **Quản lý cache:** Xóa cache khi cần tải lại dữ liệu mới
                """)
        else:
            st.warning("⚠️ Không có dữ liệu để hiển thị. Vui lòng thử khu vực khác.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 12px;">
        <p>📡 Dữ liệu từ OpenStreetMap | ⚡ Tốc độ tối ưu | 🎨 Visualized with Folium & Streamlit</p>
        <p>© 2024 Bản đồ Giao thông TP.HCM | Phiên bản 2.0</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()