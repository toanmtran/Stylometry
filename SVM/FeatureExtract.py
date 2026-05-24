import json
import re
import string
import math
import glob
import os
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Dict, Tuple, Optional

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag


nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng')

FUNCTION_WORDS = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
]
HEDGE_WORDS  = ["maybe", "perhaps", "probably", "possibly", "might", "could",
                "somewhat", "rather", "quite", "fairly", "seemingly"]
AMPLIFIERS   = ["very", "really", "extremely", "absolutely", "totally",
                "completely", "utterly", "highly", "incredibly", "remarkably"]
DISCOURSE_MARKERS = ["however", "therefore", "moreover", "furthermore",
                     "nevertheless", "meanwhile", "nonetheless", "consequently",
                     "thus", "hence", "accordingly", "besides", "otherwise"]
CONJUNCTIONS = ["and", "but", "or", "nor", "for", "yet", "so",
                "because", "although", "though", "while", "whereas", "since"]


# --- CÁC HÀM XỬ LÝ (GIỮ NGUYÊN HOẶC ĐÃ FIX POS TAGGING) ---
def _tokenize(text: str) -> Tuple[List[str], List[str], List[str]]:
    sentences = sent_tokenize(text)
    tokens_raw = word_tokenize(text)
    tokens_alpha = [t.lower() for t in tokens_raw if t.isalpha()]
    return sentences, tokens_raw, tokens_alpha

def _yule_k(tokens: List[str]) -> float:
    freq = Counter(tokens)
    freq_spectrum = Counter(freq.values())
    N = len(tokens)
    if N <= 1: return 0.0
    M2 = sum(i * i * vi for i, vi in freq_spectrum.items())
    K = 10000 * (M2 - N) / (N * N) if N > 0 else 0
    return K

def _simpson_d(tokens: List[str]) -> float:
    freq = Counter(tokens)
    N = len(tokens)
    if N <= 1: return 0.0
    D = sum(n * (n - 1) for n in freq.values()) / (N * (N - 1))
    return D

def _hapax_legomena_ratio(tokens: List[str]) -> float:
    freq = Counter(tokens)
    hapax = sum(1 for v in freq.values() if v == 1)
    return hapax / len(tokens) if tokens else 0

def _hapax_dislegomena_ratio(tokens: List[str]) -> float:
    freq = Counter(tokens)
    dis = sum(1 for v in freq.values() if v == 2)
    return dis / len(tokens) if tokens else 0

def _brunet_w(tokens: List[str]) -> float:
    N = len(tokens)
    V = len(set(tokens))
    if N <= 1 or V <= 1: return 0
    return N ** (V ** -0.172)

def _honore_r(tokens: List[str]) -> float:
    freq = Counter(tokens)
    N = len(tokens)
    V = len(freq)
    V1 = sum(1 for v in freq.values() if v == 1)
    if V1 == V or N == 0: return 0
    return 100 * math.log(N) / (1 - V1 / V) if V > 0 else 0

# (Đã fix lỗi POS context loss như tôi góp ý ở phía trên)
def _pos_tag_distribution(tokens_raw: List[str]) -> Dict[str, float]:
    if not tokens_raw: return {}
    tagged = pos_tag(tokens_raw)
    alpha_tags = [tag for word, tag in tagged if word.isalpha()]
    total = len(alpha_tags) or 1
    tag_counts = Counter(alpha_tags)
    return {tag: count / total for tag, count in tag_counts.items()}

def extract_features(text: str) -> Dict[str, float]:
    sentences, tokens_raw, tokens = _tokenize(text)
    n_sent = len(sentences) or 1
    n_tok  = len(tokens) or 1
    vocab  = set(tokens)
    n_vocab = len(vocab) or 1
    freq   = Counter(tokens)
    features: Dict[str, float] = {}

    features["n_tokens"]          = n_tok
    features["n_sentences"]       = n_sent
    features["n_vocab"]           = n_vocab
    features["type_token_ratio"]  = n_vocab / n_tok

    word_lens = [len(w) for w in tokens]
    features["avg_word_len"]      = float(np.mean(word_lens)) if word_lens else 0
    features["std_word_len"]      = float(np.std(word_lens)) if word_lens else 0
    features["median_word_len"]   = float(np.median(word_lens)) if word_lens else 0
    features["max_word_len"]      = float(max(word_lens)) if word_lens else 0
    for k in [1, 2, 3, 4, 5, 6]:
        features[f"word_len_{k}_frac"] = sum(1 for l in word_lens if l == k) / n_tok

    sent_lens = [len(word_tokenize(s)) for s in sentences]
    features["avg_sent_len"]      = float(np.mean(sent_lens)) if sent_lens else 0
    features["std_sent_len"]      = float(np.std(sent_lens)) if sent_lens else 0
    features["median_sent_len"]   = float(np.median(sent_lens)) if sent_lens else 0
    features["max_sent_len"]      = float(max(sent_lens)) if sent_lens else 0
    features["min_sent_len"]      = float(min(sent_lens)) if sent_lens else 0

    features["hapax_ratio"]       = _hapax_legomena_ratio(tokens)
    features["hapax_dis_ratio"]   = _hapax_dislegomena_ratio(tokens)
    features["yule_k"]            = _yule_k(tokens)
    features["simpson_d"]         = _simpson_d(tokens)
    features["brunet_w"]          = _brunet_w(tokens)
    features["honore_r"]          = _honore_r(tokens)

    for fw in FUNCTION_WORDS:
        features[f"fw_{fw}"] = freq.get(fw, 0) / n_tok

    for name, wordlist in [("hedge", HEDGE_WORDS), ("amplifier", AMPLIFIERS),
                           ("discourse", DISCOURSE_MARKERS), ("conj", CONJUNCTIONS)]:
        features[f"cat_{name}_rate"] = sum(freq.get(w, 0) for w in wordlist) / n_tok

    punct_counts = Counter(c for c in text if c in string.punctuation)
    total_chars = len(text) or 1
    for p, label in [(",", "comma"), (".", "period"), (";", "semicolon"),
                     (":", "colon"), ("!", "excl"), ("?", "question"),
                     ("-", "dash"), ("(", "paren"), ("'", "apost"),
                     ('"', "dquote")]:
        features[f"punct_{label}_rate"] = punct_counts.get(p, 0) / n_tok

    contraction_pattern = re.compile(r"\b\w+n[''`]t\b|\b\w+[''`](s|re|ve|ll|d|m)\b", re.I)
    contractions = contraction_pattern.findall(text)
    features["contraction_rate"] = len(contractions) / n_tok

    sent_starters = [word_tokenize(s)[0].lower() for s in sentences if word_tokenize(s)]
    starter_freq = Counter(sent_starters)
    features["pronoun_start_rate"] = sum(starter_freq.get(p, 0)
        for p in ["i", "you", "he", "she", "it", "we", "they"]) / n_sent

    n_paragraphs = max(1, len([p for p in text.split("\n\n") if p.strip()]))
    features["avg_paragraph_len"] = n_tok / n_paragraphs

    pos_dist = _pos_tag_distribution(tokens_raw)
    pos_groups = {
        "pos_noun": ["NN", "NNS", "NNP", "NNPS"],
        "pos_verb": ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
        "pos_adj":  ["JJ", "JJR", "JJS"],
        "pos_adv":  ["RB", "RBR", "RBS"],
        "pos_pron": ["PRP", "PRP$", "WP", "WP$"],
        "pos_det":  ["DT", "PDT", "WDT"],
        "pos_prep": ["IN"],
        "pos_conj": ["CC"],
        "pos_modal":["MD"],
    }
    for group_name, tags in pos_groups.items():
        features[group_name] = sum(pos_dist.get(t, 0) for t in tags)

    alpha_chars = [c.lower() for c in text if c.isalpha()]
    n_alpha = len(alpha_chars) or 1
    features["uppercase_ratio"] = sum(1 for c in text if c.isupper()) / total_chars
    for vowel in "aeiou":
        features[f"char_{vowel}_rate"] = sum(1 for c in alpha_chars if c == vowel) / n_alpha

    return features

def extract_features_batch(docs: List[str], 
                           labels: Optional[List[str]] = None,
                           authors: Optional[List[str]] = None) -> pd.DataFrame: # <-- THÊM authors
    if labels is None:
        labels = [f"Doc_{i+1}" for i in range(len(docs))]
    records = []
    
    # Dùng enumerate để lấy index (i) phục vụ cho việc khớp tên tác giả
    for i, (doc, label) in enumerate(zip(docs, labels)):
        feats = extract_features(doc)
        feats["_label"] = label
        
        # <-- THÊM CỘT TÁC GIẢ VÀO TỪNG DÒNG -->
        if authors and i < len(authors):
            feats["author"] = authors[i] 
            
        records.append(feats)
        
    df = pd.DataFrame(records).set_index("_label").fillna(0)
    return df

# --- QUÁ TRÌNH THỰC THI CHÍNH ---
if __name__ == "__main__":
    input_folder = 'D:\ML project\Stylometry\dataset\lesswrong_large\cleaned_35'  
    output_file = 'LesswrongLarge.csv'
    
    json_files = glob.glob(os.path.join(input_folder, '*.json'))
    
    if len(json_files) == 0:
        print(f"Không tìm thấy file .json nào trong thư mục '{input_folder}'")
    else:
        print(f"Tìm thấy {len(json_files)} file JSON. Bắt đầu xử lý...")
        
        all_docs = []
        all_labels = []
        all_authors = []
        
        for file_path in json_files:
            file_name = os.path.basename(file_path)
            print(f"  -> Đang đọc: {file_name}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for i, item in enumerate(data):
                all_docs.append(item.get('text', ''))
                
                label = f"{file_name}_{item.get('date', 'nodate')}_doc{i+1}"
                all_labels.append(label)
                
                all_authors.append(item.get('author', 'Unknown')) 
                
        print(f"Bắt đầu trích xuất đặc trưng cho tổng cộng {len(all_docs)} tài liệu...")
        
        df_features = extract_features_batch(all_docs, labels=all_labels, authors=all_authors)

        df_features.to_csv(output_file)
        print(f"Hoàn thành! Đã lưu toàn bộ dữ liệu vào '{output_file}'")