import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss

app = Flask(__name__)
CORS(app)

print("Veriler ve Yapay Zeka Modeli Yükleniyor... Lütfen bekleyin.")

# 1. Veriyi Yükle
with open('kararlar.json', 'r', encoding='utf-8') as f:
    kararlar = json.load(f)

# Arama yapılacak metin havuzunu oluşturuyoruz (Özet ve Hüküm birleşimi)
corpus = [k['olay_ozeti'] + " " + k['hukum'] for k in kararlar]
tokenized_corpus = [doc.lower().split() for doc in corpus]

# 2. BM25 Hazırlığı (Lexical Arama)
bm25 = BM25Okapi(tokenized_corpus)

# 3. FAISS ve Model Hazırlığı (Dense Arama)
# Demo hızlı çalışsın diye hafif bir model koydum. 
# İleride kendi eğittiğin modelin klasör yolunu buraya yazabilirsin.
model = SentenceTransformer('./BERTurk-Legal_FULL_seed42_ep2_msl192')
#model = SentenceTransformer('all-MiniLM-L6-v2') 

corpus_embeddings = model.encode(corpus)
embedding_dim = corpus_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(np.array(corpus_embeddings))

# Min-Max Normalizasyon Fonksiyonu
def normalize_scores(scores):
    min_val, max_val = np.min(scores), np.max(scores)
    if max_val == min_val:
        return [0.5 for _ in scores]
    return [(s - min_val) / (max_val - min_val) for s in scores]

import re
import numpy as np

def xai_cumle_bul(query_emb, text, model, top_k=2):
    # 1. Hukuk metinlerine özel bölücü: Nokta, ünlem, soru işareti, NOKTALI VİRGÜL veya YENİ SATIR
    cumleler = re.split(r'(?<=[.!?;\n])\s+', text)
    # Çok kısa anlamsız parçaları ele (örn: "Karar verildi.")
    cumleler = [c.strip() for c in cumleler if len(c.strip()) > 20]
    
    if not cumleler:
        return ["Açıklayıcı metin bulunamadı."]
        
    # 2. Tüm parçaları vektöre çevir
    cumle_embs = model.encode(cumleler)
    
    # 3. Cosine Similarity hesapla
    query_emb_flat = query_emb.flatten()
    skorlar = []
    for emb in cumle_embs:
        skor = np.dot(query_emb_flat, emb) / (np.linalg.norm(query_emb_flat) * np.linalg.norm(emb))
        skorlar.append(skor)
        
    # 4. En yüksek skorlu indeksleri bul
    en_iyi_idx = np.argsort(skorlar)[-top_k:][::-1]
    
    # 5. ARAYÜZ (UI) KORUMA KATMANI: Çok uzunsa kırp!
    sonuclar = []
    for i in en_iyi_idx:
        secilen_cumle = cumleler[i]
        kelimeler = secilen_cumle.split()
        
        # Eğer cümle 35 kelimeden uzunsa, 35'te kes ve sonuna üç nokta koy
        if len(kelimeler) > 35:
            secilen_cumle = " ".join(kelimeler[:35]) + "..."
            
        sonuclar.append(secilen_cumle)
        
    return sonuclar
@app.route('/api/search', methods=['GET'])
def search():
    query = request.args.get('q', '')
    
    # Kullanıcı hiçbir şey yazmadan ara derse hepsini getir
    if not query:
        return jsonify(kararlar)

    # --- 1. BM25 SKORLAMASI ---
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    norm_bm25 = normalize_scores(bm25_scores)

    # --- 2. FAISS SKORLAMASI ---
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), len(corpus))
    
    # FAISS mesafe ölçer (düşük değer = yüksek benzerlik). Bunu skora çeviriyoruz.
    faiss_scores = np.zeros(len(corpus))
    for rank, doc_id in enumerate(I[0]):
        faiss_scores[doc_id] = 1.0 / (1.0 + float(D[0][rank]))
    
    norm_faiss = normalize_scores(faiss_scores)

    # --- 3. LATE FUSION (Hibrit Harmanlama) ---
    alpha = 0.5 # Ağırlık (0.5 = İkisi de eşit derecede önemli)
    final_scores = []
    
    for i in range(len(corpus)):
        hybrid_score = (alpha * norm_bm25[i]) + ((1 - alpha) * norm_faiss[i])
        final_scores.append((hybrid_score, kararlar[i]))

    # Skorlara göre büyükten küçüğe sırala
    final_scores.sort(key=lambda x: x[0], reverse=True)

    # Vue.js'e göndermek için JSON formatını hazırla
    results = []
    for score, karar in final_scores[:10]: # Sadece ilk 10 sonuç için XAI hesapla (hız için)
        karar_kopyasi = karar.copy()
        karar_kopyasi['hibrit_skor'] = round(float(score), 2)
        
        # --- XAI MOTORU DEVREYE GİRİYOR ---
        # Kararın gerekçesini ve olay özetini birleştirip içinde arıyoruz
        hedef_metin = karar_kopyasi.get('gerekce', '') + " " + karar_kopyasi.get('tam_olay', '')
        
        # En iyi 2 cümleyi bul (query_embedding zaten FAISS aşamasından elimizde var)
        vurgular = xai_cumle_bul(np.array(query_embedding), hedef_metin, model)
        karar_kopyasi['xai_vurgular'] = vurgular
        
        results.append(karar_kopyasi)

    return jsonify(results)

if __name__ == '__main__':
    print("Sistem Hazır! Frontend'den arama yapabilirsin.")
    app.run(debug=True, port=5000)
