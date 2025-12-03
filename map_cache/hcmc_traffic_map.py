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
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Cấu hình OSMnx
ox.settings.timeout = 1000  # Tăng thời gian chờ cho các khu vực lớn
ox.settings.use_cache = True
ox.settings.log_console = False

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

# Biến toàn cục để cache trong bộ nhớ (tránh đọc file nhiều lần)
_MEMORY_CACHE = {}
_PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL  # Sử dụng protocol cao nhất cho tốc độ tốt nhất

# Hằng số cho tính toán nhanh
_EARTH_RADIUS = 6371000  # Bán kính Trái Đất (mét)
_DEG_TO_RAD = math.pi / 180.0  # Chuyển đổi độ sang radian


@lru_cache(maxsize=1000)
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Tính khoảng cách Haversine với caching để tăng tốc độ
    Sử dụng lru_cache để tránh tính toán lại các cặp tọa độ giống nhau
    """
    # Chuyển đổi độ sang radian
    lat1_rad = lat1 * _DEG_TO_RAD
    lon1_rad = lon1 * _DEG_TO_RAD
    lat2_rad = lat2 * _DEG_TO_RAD
    lon2_rad = lon2 * _DEG_TO_RAD

    # Chênh lệch tọa độ
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Công thức Haversine tối ưu
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return _EARTH_RADIUS * c


def calculate_route_length_fast(coords):
    """
    Tính chiều dài thực tế của tuyến đường từ tọa độ (tính bằng mét)
    Phiên bản tối ưu hóa tốc độ
    """
    if len(coords) < 2:
        return 0.0

    total_distance = 0.0

    # Tính khoảng cách giữa các điểm liên tiếp
    prev_lat, prev_lon = coords[0]
    for i in range(1, len(coords)):
        curr_lat, curr_lon = coords[i]
        total_distance += haversine_distance(prev_lat, prev_lon, curr_lat, curr_lon)
        prev_lat, prev_lon = curr_lat, curr_lon

    return total_distance


def calculate_total_length_parallel(edges, max_workers=4):
    """
    Tính tổng chiều dài của tất cả các tuyến đường sử dụng parallel processing
    """
    if len(edges) == 0:
        return 0.0

    # Chuyển đổi edges thành danh sách để xử lý song song
    edges_list = []
    for idx, row in edges.iterrows():
        if hasattr(row.geometry, 'coords'):
            try:
                coords = [(lat, lon) for lon, lat in row.geometry.coords]
                if len(coords) >= 2:
                    edges_list.append(coords)
            except:
                continue

    if not edges_list:
        return 0.0

    # Sử dụng ThreadPoolExecutor để tính toán song song
    total_length_m = 0.0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Gửi các task tính toán
        future_to_coords = {executor.submit(calculate_route_length_fast, coords): coords for coords in edges_list}

        # Thu thập kết quả
        for future in as_completed(future_to_coords):
            try:
                total_length_m += future.result()
            except Exception:
                continue

    return total_length_m / 1000  # Chuyển sang km


class CacheManager:
    """Quản lý cache cho ứng dụng với tối ưu hóa tốc độ"""

    @staticmethod
    def get_cache_key(place_name, detailed=False):
        """Tạo key cache từ tên địa điểm"""
        cache_string = f"{place_name}_{detailed}"
        return hashlib.md5(cache_string.encode()).hexdigest()

    @staticmethod
    def get_cache_info_path():
        """Lấy đường dẫn file thông tin cache"""
        return os.path.join(CACHE_DIR, "cache_info.json")

    @staticmethod
    def get_cache_file_path(cache_key, compressed=True):
        """Lấy đường dẫn file cache dữ liệu"""
        if compressed:
            return os.path.join(CACHE_DIR, f"{cache_key}.pkl.gz")
        else:
            return os.path.join(CACHE_DIR, f"{cache_key}.pkl")

    @staticmethod
    def get_metadata_file_path(cache_key):
        """Lấy đường dẫn file metadata"""
        return os.path.join(CACHE_DIR, f"{cache_key}_meta.json")

    @staticmethod
    def load_cache_info():
        """Tải thông tin cache từ file"""
        info_path = CacheManager.get_cache_info_path()
        if os.path.exists(info_path):
            try:
                with open(info_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    @staticmethod
    def save_cache_info(cache_info):
        """Lưu thông tin cache vào file"""
        info_path = CacheManager.get_cache_info_path()
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(cache_info, f, ensure_ascii=False, indent=2)

    @staticmethod
    def is_cache_valid(cache_key, max_age_days=30):
        """Kiểm tra cache còn hợp lệ không"""
        meta_path = CacheManager.get_metadata_file_path(cache_key)
        if not os.path.exists(meta_path):
            return False

        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Kiểm tra thời gian tạo
            created_time = datetime.fromisoformat(metadata.get('created_at', '2000-01-01'))
            age = datetime.now() - created_time

            return age.days < max_age_days
        except:
            return False

    @staticmethod
    def update_cache_metadata(cache_key, place_name, edges_count, total_length_km, detailed=False, compressed=True):
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
            'compressed': compressed
        }

        meta_path = CacheManager.get_metadata_file_path(cache_key)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        # Cập nhật cache info
        cache_info = CacheManager.load_cache_info()
        cache_info[cache_key] = {
            'name': place_name,
            'detailed': detailed,
            'count': edges_count,
            'total_length_km': total_length_km,
            'created': metadata['created_at'],
            'size_kb': metadata['size_kb'],
            'compressed': compressed
        }
        CacheManager.save_cache_info(cache_info)

    @staticmethod
    def save_cache_data(cache_key, edges, compressed=True):
        """Lưu dữ liệu cache với tối ưu hóa"""
        cache_file_path = CacheManager.get_cache_file_path(cache_key, compressed)

        try:
            if compressed:
                # Sử dụng gzip để nén dữ liệu
                with gzip.open(cache_file_path, 'wb') as f:
                    pickle.dump(edges, f, protocol=_PICKLE_PROTOCOL)
            else:
                # Lưu không nén
                with open(cache_file_path, 'wb') as f:
                    pickle.dump(edges, f, protocol=_PICKLE_PROTOCOL)

            return True
        except Exception as e:
            st.warning(f"⚠️ Lỗi khi lưu cache: {e}")
            return False

    @staticmethod
    def load_cache_data(cache_key, compressed=True):
        """Tải dữ liệu cache với tối ưu hóa"""
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


def get_graph_data(place_name, detailed=False):
    """Lấy dữ liệu đồ thị từ cache hoặc OSM - Phiên bản tối ưu hóa"""

    cache_key = CacheManager.get_cache_key(place_name, detailed)
    compressed = True  # Sử dụng nén mặc định

    # 1. Kiểm tra cache trong bộ nhớ
    if cache_key in _MEMORY_CACHE:
        edges, metadata = _MEMORY_CACHE[cache_key]
        st.info(f"⚡ Đang tải từ bộ nhớ: {metadata['edges_count']} tuyến đường")
        return edges

    # 2. Kiểm tra cache trên đĩa
    if CacheManager.is_cache_valid(cache_key):
        try:
            with st.spinner("🚀 Đang đọc dữ liệu từ cache (nhanh)..."):
                # Đọc dữ liệu từ file cache
                edges = CacheManager.load_cache_data(cache_key, compressed)

                if edges is not None:
                    # Đọc metadata
                    meta_path = CacheManager.get_metadata_file_path(cache_key)
                    if os.path.exists(meta_path):
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                    else:
                        # Tính toán lại tổng chiều dài nếu metadata không có
                        total_length_km = calculate_total_length_parallel(edges)
                        metadata = {
                            'edges_count': len(edges),
                            'total_length_km': total_length_km
                        }

                    # Lưu vào cache bộ nhớ
                    _MEMORY_CACHE[cache_key] = (edges, metadata)

                    st.success(
                        f"✅ Đã tải từ cache: {len(edges)} tuyến đường (kích thước: {metadata.get('size_kb', 0):.1f} KB)")
                    return edges
        except Exception as e:
            st.warning(f"⚠️ Lỗi khi đọc cache: {e}. Đang tải mới từ internet...")

    # 3. Nếu không có cache hợp lệ, tải từ OSM
    return download_and_cache_data(place_name, detailed, cache_key, compressed)


def download_and_cache_data(place_name, detailed, cache_key, compressed=True):
    """Tải dữ liệu từ OSM và lưu vào cache với tối ưu hóa"""

    # Xác định custom_filter dựa trên loại khu vực và chế độ chi tiết
    custom_filter = None

    # Nếu là chế độ chi tiết (đặc biệt cho Quận 1)
    if detailed:
        # Lấy tất cả các loại đường, bao gồm cả đường nhỏ
        custom_filter = '["highway"~"motorway|trunk|primary|secondary|tertiary|residential|service|living_street|unclassified"]'
        st.info("🔍 Đang tải chi tiết: Lấy cả đường nhỏ (hẻm, ngõ)...")
    elif "Ho Chi Minh City" in place_name and "District" not in place_name and "Quận" not in place_name:
        # Toàn thành phố - chỉ lấy đường chính
        custom_filter = '["highway"~"motorway|trunk|primary|secondary"]'
    else:
        # Các quận khác - lấy đường chính và đường phụ
        custom_filter = '["highway"~"motorway|trunk|primary|secondary|tertiary"]'

    try:
        with st.spinner(f"🌐 Đang tải dữ liệu từ OpenStreetMap..."):
            if custom_filter:
                G = ox.graph_from_place(
                    place_name,
                    network_type='drive',
                    simplify=True,
                    custom_filter=custom_filter
                )
            else:
                G = ox.graph_from_place(
                    place_name,
                    network_type='drive',
                    simplify=True
                )

        nodes, edges = ox.graph_to_gdfs(G)

        # Tính tổng chiều dài các tuyến đường với parallel processing
        with st.spinner("📏 Đang tính toán chiều dài đường..."):
            total_length_km = calculate_total_length_parallel(edges)

        # Lưu vào cache
        if CacheManager.save_cache_data(cache_key, edges, compressed):
            # Cập nhật metadata
            CacheManager.update_cache_metadata(cache_key, place_name, len(edges), total_length_km, detailed, compressed)

            # Lưu vào cache bộ nhớ
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

            st.success(
                f"💾 Đã lưu cache: {len(edges)} tuyến đường, {total_length_km:.1f} km (kích thước: {metadata['size_kb']:.1f} KB)")

        return edges

    except Exception as e:
        st.error(f"❌ Lỗi khi tải dữ liệu: {e}")
        return None


class HCMTrafficMap:
    def __init__(self):
        self.cache_info = CacheManager.load_cache_info()

    def create_sidebar(self):
        st.sidebar.title("⚙️ Tùy Chọn")

        # Hiển thị thông tin cache
        self.display_cache_info()

        # Thêm nút xóa cache
        st.sidebar.markdown("---")
        col1, col2 = st.sidebar.columns(2)

        with col1:
            if st.button("🗑️ Xóa tất cả cache", help="Xóa tất cả dữ liệu đã lưu để tải lại từ đầu"):
                self.clear_all_cache()

        with col2:
            if st.button("🗑️ Xóa cache Quận 1", help="Chỉ xóa cache của Quận 1"):
                self.clear_district1_cache()

        # Thêm tùy chọn chi tiết cho Quận 1
        st.sidebar.markdown("---")
        self.detailed_mode = False

        # Tạo danh sách lựa chọn + Mục tùy chỉnh
        options = list(DISTRICTS.keys()) + ["🔍 Nhập địa điểm tùy chỉnh..."]

        selection = st.sidebar.selectbox(
            "Chọn khu vực:",
            options,
            index=0
        )

        # Kiểm tra nếu chọn Quận 1 thì hiển thị tùy chọn chi tiết
        if selection == "Quận 1":
            self.detailed_mode = st.sidebar.checkbox(
                "🔎 Chế độ chi tiết (hiển thị cả hẻm, ngõ)",
                value=True,
                help="Bật để hiển thị tất cả các tuyến đường, bao gồm cả đường nhỏ trong Quận 1"
            )
            if self.detailed_mode:
                st.sidebar.caption("⚠️ Chế độ chi tiết có thể tải chậm hơn do có nhiều đường")

        # Xử lý logic chọn
        if selection == "🔍 Nhập địa điểm tùy chỉnh...":
            st.sidebar.markdown("---")
            custom_input = st.sidebar.text_input(
                "Gõ tên địa điểm (VD: Thu Duc City, Sân bay Tân Sơn Nhất):",
                "Ben Thanh Market"
            )

            display_name = custom_input

            # Tự động thêm context để tìm kiếm chính xác hơn
            place_query = custom_input

            input_lower = custom_input.lower()
            if "vietnam" not in input_lower and "hcmc" not in input_lower and "hồ chí minh" not in input_lower:
                place_query = custom_input + ", Ho Chi Minh City, Vietnam"
                st.sidebar.caption("Đã tự động thêm `, Ho Chi Minh City, Vietnam` vào tìm kiếm.")

            return place_query, display_name, self.detailed_mode

        else:
            # Chọn từ menu cố định
            return DISTRICTS[selection], selection, self.detailed_mode

    def display_cache_info(self):
        """Hiển thị thông tin cache trong sidebar"""
        total_size = sum(info.get('size_kb', 0) for info in self.cache_info.values())
        total_length = sum(info.get('total_length_km', 0) for info in self.cache_info.values())
        compressed_count = sum(1 for info in self.cache_info.values() if info.get('compressed', False))

        st.sidebar.markdown(f"### 📊 Thông tin Cache")
        st.sidebar.markdown(f"**Số khu vực:** {len(self.cache_info)}")
        st.sidebar.markdown(f"**Đã nén:** {compressed_count}/{len(self.cache_info)}")
        st.sidebar.markdown(f"**Tổng dung lượng:** {total_size:.1f} KB")
        st.sidebar.markdown(f"**Tổng chiều dài:** {total_length:.1f} km")

        # Hiển thị danh sách cache
        if self.cache_info:
            st.sidebar.markdown("**Top 5 cache lớn nhất:**")
            # Sắp xếp theo kích thước
            sorted_cache = sorted(self.cache_info.items(),
                                  key=lambda x: x[1].get('size_kb', 0),
                                  reverse=True)[:5]

            for cache_key, info in sorted_cache:
                name = info.get('name', 'Unknown')[:20] + "..." if len(info.get('name', '')) > 20 else info.get('name',
                                                                                                                'Unknown')
                count = info.get('count', 0)
                length = info.get('total_length_km', 0)
                size = info.get('size_kb', 0)
                compressed = "✓" if info.get('compressed', False) else "✗"
                st.sidebar.caption(f"• {name}: {count} đường, {length:.1f} km, {size:.1f} KB [{compressed}]")

            if len(self.cache_info) > 5:
                st.sidebar.caption(f"... và {len(self.cache_info) - 5} khu vực khác")

    def clear_all_cache(self):
        """Xóa tất cả file cache trong thư mục cache"""
        try:
            # Xóa cache trong bộ nhớ
            global _MEMORY_CACHE
            _MEMORY_CACHE.clear()

            # Xóa file cache
            cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith(('.pkl', '.json', '.gz'))]
            deleted_count = 0

            for file in cache_files:
                try:
                    os.remove(os.path.join(CACHE_DIR, file))
                    deleted_count += 1
                except:
                    pass

            # Xóa cache info
            CacheManager.save_cache_info({})

            st.sidebar.success(f"✅ Đã xóa {deleted_count} file cache")
            st.rerun()

        except Exception as e:
            st.sidebar.error(f"❌ Lỗi khi xóa cache: {e}")

    def clear_district1_cache(self):
        """Xóa cache của Quận 1 (cả chế độ thường và chi tiết)"""
        try:
            # Xác định các key cache của Quận 1
            district1_normal = CacheManager.get_cache_key("District 1, Ho Chi Minh City, Vietnam", detailed=False)
            district1_detailed = CacheManager.get_cache_key("District 1, Ho Chi Minh City, Vietnam", detailed=True)

            # Xóa từ cache bộ nhớ
            global _MEMORY_CACHE
            for key in [district1_normal, district1_detailed]:
                if key in _MEMORY_CACHE:
                    del _MEMORY_CACHE[key]

            # Xóa file cache
            cache_files = os.listdir(CACHE_DIR)
            deleted_count = 0

            for file in cache_files:
                file_path = os.path.join(CACHE_DIR, file)
                if file.endswith(('.pkl', '.json', '.gz')):
                    # Kiểm tra nếu file thuộc cache Quận 1
                    if district1_normal in file or district1_detailed in file:
                        try:
                            os.remove(file_path)
                            deleted_count += 1
                        except:
                            pass

            # Cập nhật cache info
            cache_info = CacheManager.load_cache_info()
            for key in [district1_normal, district1_detailed]:
                if key in cache_info:
                    del cache_info[key]
            CacheManager.save_cache_info(cache_info)

            st.sidebar.success(f"✅ Đã xóa {deleted_count} file cache Quận 1")
            st.rerun()

        except Exception as e:
            st.sidebar.error(f"❌ Lỗi khi xóa cache Quận 1: {e}")

    def load_data(self, place_query, display_name, detailed=False):
        try:
            with st.spinner(f"🚀 Đang tải dữ liệu: {display_name}..."):
                edges = get_graph_data(place_query, detailed)

            if edges is not None:
                # Lấy metadata từ cache bộ nhớ
                cache_key = CacheManager.get_cache_key(place_query, detailed)
                if cache_key in _MEMORY_CACHE:
                    edges_data, metadata = _MEMORY_CACHE[cache_key]
                    total_length_km = metadata.get('total_length_km', 0)
                else:
                    # Tính tổng chiều dài nếu không có trong cache
                    with st.spinner("📏 Đang tính toán chiều dài..."):
                        total_length_km = calculate_total_length_parallel(edges)

                # Thống kê số lượng đường theo loại
                if not edges.empty and 'highway' in edges.columns:
                    highway_counts = edges['highway'].value_counts()

                    # Hiển thị thống kê trong sidebar
                    st.sidebar.markdown("---")
                    st.sidebar.markdown("### 📈 Thống kê đường")

                    # Lấy top 10 loại đường phổ biến nhất
                    top_highways = highway_counts.head(10)
                    for hw_type, count in top_highways.items():
                        # Xử lý trường hợp highway là list
                        if isinstance(hw_type, list):
                            hw_type = ', '.join(hw_type)
                        st.sidebar.caption(f"• {hw_type}: {count} đường")

                    st.sidebar.caption(f"**Tổng:** {len(edges)} tuyến đường, {total_length_km:.1f} km")

                st.success(f"✅ Đã tải: {display_name} ({len(edges)} tuyến đường, {total_length_km:.1f} km)")
                if detailed:
                    st.info(f"🔍 Đang ở chế độ chi tiết")

                # Lưu tổng chiều dài để sử dụng sau
                edges.attrs['total_length_km'] = total_length_km

            return edges

        except Exception as e:
            st.error(f"❌ Không tìm thấy địa điểm '{display_name}'!")
            st.info(f"💡 Lỗi chi tiết: {e}")
            st.info("💡 Lỗi này xảy ra khi OpenStreetMap không nhận ra tên bạn gõ. Hãy thử gõ tiếng Anh không dấu nhé!")
            return None

    def create_map(self, edges):
        # Tính tâm bản đồ
        if not edges.empty:
            bounds = edges.total_bounds
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lon = (bounds[0] + bounds[2]) / 2
        else:
            # Tọa độ mặc định: Quận 1
            center_lat, center_lon = 10.7769, 106.7009

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=15 if len(edges) > 1000 else 14,
            tiles='OpenStreetMap',
            prefer_canvas=True
        )

        # MÀU DUY NHẤT CHO TẤT CẢ CÁC LOẠI ĐƯỜNG
        UNIFIED_COLOR = '#3388ff'  # Màu xanh dương

        count = 0
        max_edges = 50000  # Tăng giới hạn cho chế độ chi tiết
        total_displayed_length = 0.0

        # Vẽ các tuyến đường với tối ưu hóa
        for idx, row in edges.iterrows():
            if count > max_edges:
                break
            try:
                hw = row.get('highway')
                if isinstance(hw, list):
                    hw = hw[0]

                # Điều chỉnh độ dày đường dựa trên loại đường
                if hw in ['motorway', 'trunk', 'primary']:
                    weight = 3.5
                elif hw in ['secondary']:
                    weight = 2.5
                elif hw in ['tertiary']:
                    weight = 2.0
                elif hw in ['residential', 'living_street', 'unclassified']:
                    weight = 1.5
                elif hw in ['service']:
                    weight = 1.0
                else:
                    weight = 1.5

                if hasattr(row.geometry, 'coords'):
                    # Tính chiều dài thực tế của tuyến đường
                    coords = [(lat, lon) for lon, lat in row.geometry.coords]
                    route_length_m = calculate_route_length_fast(coords)
                    total_displayed_length += route_length_m

                    # Format chiều dài hiển thị
                    if route_length_m >= 1000:
                        length_display = f"{route_length_m / 1000:.2f} km"
                    else:
                        length_display = f"{route_length_m:.0f} m"

                    # Tạo popup với thông tin chi tiết
                    popup_text = f"""
                    <div style="font-family: Arial; font-size: 12px;">
                        <b>Tên đường:</b> {row.get('name', 'Không có tên')}<br>
                        <b>Loại đường:</b> {hw}<br>
                        <b>Chiều dài:</b> {length_display}<br>
                        <b>Số điểm:</b> {len(coords)}
                    </div>
                    """

                    folium.PolyLine(
                        locations=coords,
                        color=UNIFIED_COLOR,
                        weight=weight,
                        opacity=0.8,
                        popup=folium.Popup(popup_text, max_width=300),
                        tooltip=f"{row.get('name', 'Đường không tên')} - {length_display}"
                    ).add_to(m)
                    count += 1

                    # Tối ưu: vẽ theo batch nếu có nhiều đường
                    if count % 1000 == 0:
                        st.text(f"Đã vẽ {count} đường...")
            except Exception:
                continue

        # Thêm marker cho trung tâm thành phố nếu là Quận 1
        if "District 1" in str(edges.crs) if edges.crs else False:
            # Thêm các địa điểm nổi tiếng ở Quận 1
            landmarks = [
                ("Chợ Bến Thành", 10.772, 106.698),
                ("Nhà hát Thành phố", 10.777, 106.703),
                ("Bưu điện Trung tâm", 10.780, 106.699),
                ("Dinh Độc Lập", 10.777, 106.695),
                ("Bến Bạch Đằng", 10.773, 106.706)
            ]

            for name, lat, lon in landmarks:
                folium.Marker(
                    location=[lat, lon],
                    popup=name,
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(m)

        # Lưu thông tin chiều dài đã hiển thị
        m.total_displayed_length_km = total_displayed_length / 1000
        m.total_displayed_edges = count

        return m


def main():
    st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🗺️ BẢN ĐỒ GIAO THÔNG TP.HCM</h1>",
                unsafe_allow_html=True)

    # Thông tin phiên bản tối ưu
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🚀 Phiên bản tối ưu hóa**")
    st.sidebar.caption("• Cache nén GZIP")
    st.sidebar.caption("• Parallel processing")
    st.sidebar.caption("• Memory caching")

    app = HCMTrafficMap()

    # 1. Menu chọn
    place_query, display_name, detailed_mode = app.create_sidebar()

    # 2. Tải & Vẽ
    if place_query:
        edges = app.load_data(place_query, display_name, detailed_mode)
        if edges is not None:
            traffic_map = app.create_map(edges)
            st_folium(traffic_map, width=1400, height=700, returned_objects=[])

            # Lấy thông tin tổng chiều dài từ edges
            total_length_km = edges.attrs.get('total_length_km', 0)
            displayed_length_km = getattr(traffic_map, 'total_displayed_length_km', 0)
            displayed_edges = getattr(traffic_map, 'total_displayed_edges', 0)

            # Cảnh báo khi đang ở chế độ tải nặng
            if "Toàn Thành Phố" in display_name:
                st.warning("⚠️ Đang xem chế độ toàn thành phố. Chỉ hiển thị các trục đường chính để tránh treo máy.")

            # Hiển thị thông tin chi tiết
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"🎨 Màu đường: xanh dương (#3388ff)")

            with col2:
                st.info(f"📏 Tổng số tuyến đường: {len(edges)}")

            with col3:
                st.info(f"📏 Tổng chiều dài: {total_length_km:.1f} km")

            # Thông tin về số lượng đã hiển thị
            if displayed_edges < len(edges):
                st.warning(
                    f"⚠️ Hiển thị {displayed_edges}/{len(edges)} tuyến đường ({displayed_length_km:.1f}/{total_length_km:.1f} km) để đảm bảo hiệu suất")

            # Thông tin đặc biệt cho Quận 1
            if display_name == "Quận 1" and detailed_mode:
                st.success(f"""
                **Quận 1 - Chế độ chi tiết:**
                - Hiển thị tất cả các loại đường: đường lớn, đường phụ, hẻm, ngõ
                - Độ dày đường được phân biệt theo loại đường
                - Các địa điểm nổi tiếng được đánh dấu bằng icon đỏ
                - Hiển thị thông tin chi tiết khi click vào từng đường (bao gồm chiều dài thực tế)
                - Tổng chiều dài đường: {total_length_km:.1f} km
                """)


if __name__ == "__main__":
    main()