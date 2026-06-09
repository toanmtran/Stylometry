import requests
import json
import re
from bs4 import BeautifulSoup
import time
import os

# CONFIG
MIN_ARTICLES = 20
MAX_ARTICLES = 40
GRAPHQL_URL = "https://www.lesswrong.com/graphql"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Content-Type": "application/json"
}

# UTILS
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

# CORE
def scrape_single_author(slug, min_words):
    articles_recent = [] # Chỉ chứa bài >= 2024
    articles_old = []    # Chỉ chứa bài <= 2022

    print(f"\n⏳ Đang xử lý author: {slug}")

    # 1. Get user ID 
    query_user = """
    query GetUser($slug: String) {
      user(input: {selector: {slug: $slug}}) {
        result { _id slug }
      }
    }
    """

    try:
        res = requests.post(GRAPHQL_URL, headers=HEADERS, json={"query": query_user, "variables": {"slug": slug}})
        if res.status_code != 200:
            print(f"  ❌ Lỗi Server (HTTP {res.status_code}).")
            return False, 0, 0
            
        data = res.json()
        user_data = data.get('data', {}).get('user', {}).get('result')
        if not user_data:
            print(f"  ❌ Không tìm thấy ID cho slug: '{slug}'.")
            return False, 0, 0
            
        user_id = user_data['_id']
    except Exception as e:
        print(f"  ❌ Lỗi kết nối API: {e}")
        return False, 0, 0

    # 2. Crawl posts
    offset = 0
    limit = 100
    
    # Vòng lặp sẽ tiếp tục khi ít nhất 1 trong 2 nhóm chưa đạt đủ MAX
    while len(articles_recent) < MAX_ARTICLES or len(articles_old) < MAX_ARTICLES:
        query_posts = """
        query GetPosts($userId: String, $limit: Int, $offset: Int) {
          posts(input: {terms: {view: "userPosts", userId: $userId, limit: $limit, offset: $offset}}) {
            results { postedAt htmlBody }
          }
        }
        """
        try:
            res = requests.post(GRAPHQL_URL, headers=HEADERS, json={
                "query": query_posts,
                "variables": {"userId": user_id, "limit": limit, "offset": offset}
            })
            if res.status_code != 200: break
            batch = res.json()['data']['posts']['results']
        except:
            break

        if not batch:  # Hết bài của tác giả này
            break

        for post in batch:
            # Nếu cả 2 nhóm đều đã chạm mốc 40, dừng duyệt thêm
            if len(articles_recent) >= MAX_ARTICLES and len(articles_old) >= MAX_ARTICLES:
                break

            date_str = post.get('postedAt', '')[:10]
            if not date_str: continue
            try: year = int(date_str[:4])
            except: continue

            # kiểm tra điều kiện từng nhóm
            if year >= 2024 and len(articles_recent) < MAX_ARTICLES:
                target_list = articles_recent
                label = ">=2024"
            elif year <= 2022 and len(articles_old) < MAX_ARTICLES:
                target_list = articles_old
                label = "<=2022"
            else: # Bỏ qua nếu không đạt yêu cầu
                continue

            clean_text = clean_html(post.get('htmlBody', ''))
            word_count = count_words_accurately(clean_text)
            
            if word_count < min_words: 
                continue

            # Đưa bài vào danh sách
            target_list.append({
                "author": slug,
                "text": clean_text,
                "category": "blog",
                "platform": "lesswrong",
                "date": date_str,
            })
            
            # In tiến độ từng nhóm
            print(f"    ✅ Đã lấy 1 bài {label} | Năm: {year} | Số từ: {word_count}")
            print(f"        Tiến độ: [>=2024: {len(articles_recent)}/{MAX_ARTICLES} bài] --- [<=2022: {len(articles_old)}/{MAX_ARTICLES} bài]")

        offset += limit
        time.sleep(0.5)

    # 3. Đánh giá kết quả
    # thỏa mãn: nhóm recent >= 20 VÀ nhóm old >= 20
    if len(articles_recent) >= MIN_ARTICLES and len(articles_old) >= MIN_ARTICLES:
        all_articles = articles_recent + articles_old
        filename = f"lesswrong_{slug}_data.json"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        target_dir = os.path.join(os.path.dirname(script_dir), "raw", "raw_10")
        os.makedirs(target_dir, exist_ok=True)
        file_path = os.path.join(target_dir, filename)
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(all_articles, f, indent=2, ensure_ascii=False)
        
        print(f"\n  🎉 THÀNH CÔNG: Đã lưu tổng cộng {len(all_articles)} bài viết (Trong đó: {len(articles_recent)} bài mới, {len(articles_old)} bài cũ)")
        return True, len(articles_recent), len(articles_old)
    else:
        print(f"\n  ❌ THẤT BẠI: Không đạt đủ điều kiện tối thiểu 20 bài cho MỖI nhóm.")
        print(f"     Thực tế gom được: >=2024 có {len(articles_recent)} bài | <=2022 có {len(articles_old)} bài.")
        return False, len(articles_recent), len(articles_old)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.join(script_dir, 'link.json')
    
    if not os.path.exists(json_file):
        print(f"❌ Không tìm thấy file {json_file}")
        exit()

    urls = []
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            for i, item in enumerate(json_data, 1):
                u = item.get(f"link{i}", "").strip()
                if u: 
                    clean_slug = u.split('/users/')[-1].split('?')[0].split('#')[0].strip('/')
                    if clean_slug not in urls:
                        urls.append(clean_slug)
    except Exception as e:
        print(f"❌ Lỗi đọc JSON: {e}")
        exit()

    if not urls:
        print("❌ Không có link hợp lệ.")
        exit()

    try:
        min_words_input = int(input(f"Tìm thấy {len(urls)} tác giả. Nhập Min words: "))
    except:
        min_words_input = 1500

    summary = []
    for slug in urls:
        ok, r, o = scrape_single_author(slug, min_words_input)
        summary.append({"slug": slug, "recent": r, "old": o, "status": "✅ ĐẠT" if ok else "❌ LOẠI"})
        time.sleep(1)

    print("\n" + "="*80)
    print("📊 KẾT QUẢ TỔNG HỢP SAU KHI CRAWL")
    print("="*80)
    print(f"{'STT':<5} | {'Author':<25} | {'Nhóm >=2024':<12} | {'Nhóm <=2022':<12} | {'Trạng thái'}")
    print("-" * 80)
    success_count = 0
    for idx, item in enumerate(summary, 1):
        if item['status'] == "✅ ĐẠT": success_count += 1
        print(f"{idx:<5} | {item['slug']:<25} | {item['recent']:<12} | {item['old']:<12} | {item['status']}")
    print("-" * 80)
    print(f"📌 Tổng số tác giả thỏa mãn điều kiện: {success_count}/{len(urls)}")