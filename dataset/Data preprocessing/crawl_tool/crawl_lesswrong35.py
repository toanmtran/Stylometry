import requests
import json
import re
from bs4 import BeautifulSoup
import time
import os

# ================= CONFIG =================
MIN_ARTICLES = 30
MAX_ARTICLES = 50
GRAPHQL_URL = "https://www.lesswrong.com/graphql"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Content-Type": "application/json"
}

# ================= UTILS =================
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

# ================= CORE =================
def scrape_single_author(slug, min_words):
    articles = []

    print(f"\n⏳ Đang xử lý author: {slug}")

    # --- 1. Get user ID ---
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
            return False, 0
            
        data = res.json()
        user_data = data.get('data', {}).get('user', {}).get('result')
        if not user_data:
            print(f"  ❌ Không tìm thấy ID cho slug: '{slug}'.")
            return False, 0
            
        user_id = user_data['_id']
    except Exception as e:
        print(f"  ❌ Lỗi kết nối API: {e}")
        return False, 0

    # --- 2. Crawl posts ---
    offset = 0
    limit = 100
    
    # Vòng lặp sẽ tiếp tục cho đến khi gom đủ MAX (50 bài)
    while len(articles) < MAX_ARTICLES:
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

        if not batch: 
            break # Hết bài viết của tác giả này

        for post in batch:
            # Nếu đã chạm mốc MAX_ARTICLES, dừng duyệt thêm
            if len(articles) >= MAX_ARTICLES:
                break

            clean_text = clean_html(post.get('htmlBody', ''))
            word_count = count_words_accurately(clean_text)
            
            if word_count < min_words: 
                continue

            date_str = post.get('postedAt', '')[:10]

            # Đưa bài vào danh sách
            articles.append({
                "author": slug,
                "text": clean_text,
                "category": "blog",
                "platform": "lesswrong",
                "date": date_str,
            })
            
            # In tiến độ
            print(f"    ✅ Đã lấy 1 bài | Số từ: {word_count} | Ngày: {date_str}")
            print(f"        Tiến độ: [{len(articles)}/{MAX_ARTICLES} bài]")

        offset += limit
        time.sleep(0.5)

    # --- 3. Đánh giá kết quả ---
    # Phải đạt đủ số bài tối thiểu
    if len(articles) >= MIN_ARTICLES:
        filename = f"lesswrong_{slug}_data.json"
        
        # --- CODE MỚI ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        target_dir = os.path.join(os.path.dirname(script_dir), "Dataset", "lesswrong35")
        os.makedirs(target_dir, exist_ok=True)
        file_path = os.path.join(target_dir, filename)
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        # ----------------
        
        print(f"\n  🎉 THÀNH CÔNG: Đã lưu tổng cộng {len(articles)} bài viết.")
        return True, len(articles)
    else:
        print(f"\n  ❌ THẤT BẠI: Không đạt đủ điều kiện tối thiểu {MIN_ARTICLES} bài.")
        print(f"     Thực tế gom được: {len(articles)} bài.")
        return False, len(articles)

# ================= MAIN =================
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
        ok, count = scrape_single_author(slug, min_words_input)
        summary.append({"slug": slug, "count": count, "status": "✅ ĐẠT" if ok else "❌ LOẠI"})
        time.sleep(1)

    print("\n" + "="*80)
    print("📊 KẾT QUẢ TỔNG HỢP SAU KHI CRAWL")
    print("="*80)
    print(f"{'STT':<5} | {'Author':<30} | {'Số bài':<10} | {'Trạng thái'}")
    print("-" * 80)
    success_count = 0
    for idx, item in enumerate(summary, 1):
        if item['status'] == "✅ ĐẠT": success_count += 1
        print(f"{idx:<5} | {item['slug']:<30} | {item['count']:<10} | {item['status']}")
    print("-" * 80)
    print(f"📌 Tổng số tác giả thỏa mãn điều kiện ({MIN_ARTICLES} bài trở lên): {success_count}/{len(urls)}")