import requests
import json
import re
from bs4 import BeautifulSoup
import time
import os

def count_words_accurately(text):
    words = re.findall(r"\b[\w'-]+\b", text)
    return len(words)

def clean_html(html_str):
    if not html_str: return ""
    soup = BeautifulSoup(html_str, 'html.parser')
    tags_to_remove = ['img', 'figure', 'figcaption', 'script', 'style', 'iframe', 'svg', 'button']
    for tag in soup.find_all(tags_to_remove): tag.decompose()
    for a_tag in soup.find_all('a'): a_tag.unwrap()
    return soup.get_text(separator=' ', strip=True)

def scrape_lesswrong_api(slug, min_words):
    print(f"\n🚀 CRAWL GRAPHQL API CHO TÁC GIẢ: {slug.upper()}")
    url = "https://www.lesswrong.com/graphql"
    
    target_new = 5    # 5 bài >= 2025
    target_old = 145  # 145 bài <= 2024
    
    # 1: Lấy User ID
    print("🔍 Đang truy tìm ID hệ thống của tác giả...")
    query_user = """
    query GetUser($slug: String) {
      user(input: {selector: {slug: $slug}}) {
        result { _id }
      }
    }
    """
    res = requests.post(url, json={"query": query_user, "variables": {"slug": slug}})
    try:
        user_id = res.json()['data']['user']['result']['_id']
        print(f"✅ Đã tìm thấy ID: {user_id}")
    except:
        print("❌ Lỗi: Không tìm thấy tác giả này. Hãy kiểm tra lại link!")
        return

    # 2: crawl phân lô từ api
    print("⚡ Đang tải lịch sử bài viết (Chia thành từng lô 100 bài)...")
    posts = []
    offset = 0
    limit = 100 
    
    while True:
        query_posts = """
        query GetPosts($userId: String, $limit: Int, $offset: Int) {
          posts(input: {terms: {view: "userPosts", userId: $userId, limit: $limit, offset: $offset}}) {
            results {
              postedAt
              htmlBody
            }
          }
        }
        """
        try:
            res = requests.post(url, json={
                "query": query_posts, 
                "variables": {"userId": user_id, "limit": limit, "offset": offset}
            })
            batch = res.json()['data']['posts']['results']
        except Exception as e:
            print(f"❌ Lỗi mạng khi tải ở lô {offset}: {e}")
            break
            
        if not batch: break 
            
        posts.extend(batch)
        print(f"  -> Đã gom được {len(posts)} bài...")
        
        if len(batch) < limit: break
            
        offset += limit
        time.sleep(0.5) # Thời gian hồi chiêu chống ban IP
        
    print(f"\n✅ Đã tải về thành công toàn bộ {len(posts)} bài viết của tác giả!\n")

    # 3: Dọn rác, Đếm từ và chia khoảng
    count_old = 0
    count_new = 0
    articles_data = []

    for post in posts:
        # Dừng lại nếu cả 2 giỏ đều đã đầy
        if count_old >= target_old and count_new >= target_new:
            print("\n🎉 ĐÃ THU THẬP ĐỦ SỐ LƯỢNG YÊU CẦU (5 bài 2025 + 145 bài <=2024)!")
            break
            
        date_str = post.get('postedAt', '')
        if not date_str: continue
        date_str = date_str[:10]
        year = int(date_str[:4])
        
        # Phân loại năm
        is_old_group = False
        if year <= 2024:
            if count_old >= target_old: 
                continue # Nếu giỏ cũ đã đủ 145 bài thì bỏ qua
            is_old_group = True
        elif year >= 2025:
            if count_new >= target_new: 
                continue # Nếu giỏ mới đã đủ 5 bài thì bỏ qua
            is_old_group = False
        else:
            continue 
            
        clean_text = clean_html(post.get('htmlBody', ''))
        word_count = count_words_accurately(clean_text)
        
        if word_count < min_words: 
            continue
        
        articles_data.append({
            "author": slug,
            "text": clean_text,
            "category": "blog",
            "platform": "lesswrong",
            "date": date_str
        })
        
        # Cập nhật bộ đếm
        if is_old_group:
            count_old += 1
            print(f"  ✅ [<= 2024] Đã lưu! (Năm {year} | {word_count} từ) - Tiến độ: {count_old}/{target_old}")
        else:
            count_new += 1
            print(f"  ✅ [>= 2025] Đã lưu! (Năm {year} | {word_count} từ) - Tiến độ: {count_new}/{target_new}")

    # 4: Xuất file JSON
    output_filename = f"lesswrong_{slug}_data.json"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(os.path.dirname(script_dir), "Dataset", "lesswrong5")
    os.makedirs(target_dir, exist_ok=True)
    file_path = os.path.join(target_dir, output_filename)
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(articles_data, f, indent=2, ensure_ascii=False)

        
    print("\n" + "="*50)
    print("📈 BÁO CÁO KẾT QUẢ THU THẬP TỪ API:")
    print(f"  🔹 Tổng số bài viết hợp lệ đã lưu : {len(articles_data)} bài")
    print(f"  🔹 Nhóm bài cũ (<= 2024)          : {count_old} / {target_old} bài mục tiêu")
    print(f"  🔹 Nhóm bài mới (>= 2025)         : {count_new} / {target_new} bài mục tiêu")
    print(f"  💾 Đã lưu vào file                : {file_path}") # Sửa lại in ra đường dẫn đầy đủ
    print("="*50 + "\n")

if __name__ == "__main__":
    print("=== CỖ MÁY CRAWL DỮ LIỆU LESSWRONG (CUSTOM BUCKETS) ===")
    user_url = input("1. Dán link tác giả (VD: https://www.lesswrong.com/users/zvi): ").strip()
    slug = user_url.split('?')[0].split('#')[0].rstrip('/').split('/')[-1]
    
    # Đã bỏ qua câu hỏi nhập tổng số bài N vì chúng ta khóa cứng mục tiêu 5 - 145.
    
    while True:
        try:
            min_words = int(input("2. Nhập số từ tối thiểu để nhận bài (VD: 1500): "))
            break
        except ValueError:
            print("Vui lòng nhập một số nguyên!")
    
    scrape_lesswrong_api(slug, min_words)