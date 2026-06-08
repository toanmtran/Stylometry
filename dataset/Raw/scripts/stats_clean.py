import json
import os
import glob

def load_json_files_from_dir(target_dir, folder_type):
    if not os.path.exists(target_dir):
        print(f"⚠️ Thư mục không tồn tại: {target_dir}")
        return []

    pattern = os.path.join(target_dir, "*.json")
    files = sorted(glob.glob(pattern))
    
    if not files: return []
    
    authors = []
    for filepath in files:
        filename = os.path.basename(filepath)
        folder_name = os.path.basename(os.path.dirname(filepath))
        # Làm sạch tên file để lấy slug chuẩn xác (loại bỏ thêm các tiền tố/hậu tố nếu có)
        slug = filename.replace("lesswrong_", "").replace("cleaned_", "").replace("_data.json", "").replace(".json", "")
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except:
            authors.append({"slug": slug, "folder": folder_name, "total": 0, "col1": 0, "col2": 0, "error": True})
            continue
        
        c1 = 0  # Cột trái (Bài mới)
        c2 = 0  # Cột phải (Bài cũ)
        
        # Chia năm đối với cleaned_5 và cleaned_10
        if folder_type in ['cleaned_5', 'cleaned_10']:
            for article in data:
                date_str = article.get("date", "")
                if not date_str or len(date_str) < 4:
                    continue
                try:
                    year = int(date_str[:4])
                    
                    if folder_type == 'cleaned_5':
                        if year >= 2025: c1 += 1
                        else: c2 += 1  # <= 2024
                    elif folder_type == 'cleaned_10':
                        if year >= 2024: c1 += 1
                        elif year <= 2022: c2 += 1
                        
                except ValueError: 
                    continue
        
        authors.append({
            "slug": slug, "folder": folder_name, "total": len(data),
            "col1": c1, "col2": c2, "error": False
        })
    return authors

def print_table(authors, folder_type):
    # Tiêu đề bảng thay đổi linh hoạt theo thư mục
    if folder_type == 'cleaned_5':
        headers = ["STT", "Thư mục", "Tác giả", "Bài >= 2025", "Bài <= 2024", "Tổng bài"]
    elif folder_type == 'cleaned_10':
        headers = ["STT", "Thư mục", "Tác giả", "Bài >= 2024", "Bài <= 2022", "Tổng bài"]
    else: # cleaned_25 và cleaned_35
        headers = ["STT", "Thư mục", "Tác giả", "Tổng bài"]
        
    widths = [len(h) for h in headers]
    rows = []
    
    # In hàng tùy theo số lượng cột của bảng
    for i, a in enumerate(authors, start=1):
        if folder_type in ['cleaned_5', 'cleaned_10']:
            row = [
                str(i), a["folder"], a["slug"], 
                str(a["col1"]) if not a["error"] else "ERR", 
                str(a["col2"]) if not a["error"] else "ERR", 
                str(a["total"]) if not a["error"] else "ERR"
            ]
        else:
            row = [
                str(i), a["folder"], a["slug"], 
                str(a["total"]) if not a["error"] else "ERR"
            ]
        rows.append(row)
        for j, cell in enumerate(row): widths[j] = max(widths[j], len(cell))
            
    # Tính tổng cộng
    if folder_type in ['cleaned_5', 'cleaned_10']:
        total_row = [
            "", "TỔNG CỘNG", "", 
            str(sum(a["col1"] for a in authors)), 
            str(sum(a["col2"] for a in authors)), 
            str(sum(a["total"] for a in authors))
        ]
    else:
        total_row = [
            "", "TỔNG CỘNG", "", 
            str(sum(a["total"] for a in authors))
        ]
        
    for j, cell in enumerate(total_row): widths[j] = max(widths[j], len(cell))
        
    def hline(left, mid, right, fill="─"): return left + mid.join([fill * (w + 2) for w in widths]) + right
    def fmt_row(cells):
        return "│" + "│".join([f" {cell:^{widths[j]}} " if j not in (1, 2) else f" {cell:<{widths[j]}} " for j, cell in enumerate(cells)]) + "│"
    
    print(hline("┌", "┬", "┐"))
    print(fmt_row(headers))
    print(hline("├", "┼", "┤"))
    for row in rows: print(fmt_row(row))
    print(hline("├", "┼", "┤"))
    print(fmt_row(total_row))
    print(hline("└", "┴", "┘"))

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(os.path.dirname(script_dir), "Dataset")
    
    print("\n" + "=" * 80)
    print("📊 MENU THỐNG KÊ DỮ LIỆU ĐÃ CLEAN (stats_clean.py)")
    print("=" * 80)
    print("1. Thống kê thư mục: cleaned_5  (Tiêu chí: >= 2025 và <= 2024)")
    print("2. Thống kê thư mục: cleaned_10 (Tiêu chí: >= 2024 và <= 2022)")
    print("3. Thống kê thư mục: cleaned_25 (Chỉ đếm tổng số bài)")
    print("4. Thống kê thư mục: cleaned_35 (Chỉ đếm tổng số bài)")
    print("-" * 80)
    
    choice = input("Nhập lựa chọn của bạn (1-4): ").strip()
    folder_map = {
        '1': 'cleaned_5',
        '2': 'cleaned_10',
        '3': 'cleaned_25',
        '4': 'cleaned_35'
    }

    folder = folder_map.get(choice)
    if not folder: 
        print("❌ Lựa chọn không hợp lệ. Vui lòng chạy lại script.")
        return
        
    target_dir = os.path.join(dataset_dir, folder)
    print(f"\n📂 Đang quét thư mục: {folder}...\n")
    authors = load_json_files_from_dir(target_dir, folder)
    
    if not authors: 
        print(f"⚠️ Không có dữ liệu hợp lệ hoặc thư mục đang trống.")
        return
    
    print_table(authors, folder)

if __name__ == "__main__":
    main()