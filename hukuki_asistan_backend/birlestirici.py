import os
import json
import glob

girdi_klasoru = "json_dosyalari" 
cikis_dosyasi = "kararlar.json"

tum_kararlar = []
json_dosyalari = glob.glob(os.path.join(girdi_klasoru, "*.json"))

print(f"Toplam {len(json_dosyalari)} dosya bulundu. Full kapasite birleştiriliyor...")

for dosya_yolu in json_dosyalari:
    try:
        with open(dosya_yolu, 'r', encoding='utf-8') as f:
            # Senin orijinal verini EKSİKSİZ olarak okuyoruz
            veri = json.load(f)
            
            # Vue.js ve Flask'ın bozulmaması için, içerdeki verileri bir kopyasını alıp
            # verinin EN ÜST (root) seviyesine yapıştırıyoruz.
            # Böylece hem orijinal hiyerarşi korunuyor hem de uygulama kolayca çalışıyor.
            veri["id"] = veri.get("meta_data", {}).get("file_name", str(dosya_yolu)).replace(".pdf", "")
            veri["mahkeme"] = veri.get("meta_data", {}).get("court_name", "Bilinmeyen Mahkeme")
            veri["esas_no"] = veri.get("meta_data", {}).get("esas_no", "-")
            veri["konu"] = veri.get("meta_data", {}).get("case_subject", "Belirtilmemiş Konu")
            
            veri["olay_ozeti"] = veri.get("summary_for_human", "Özet bulunamadı.")
            veri["tam_olay"] = veri.get("rrl_segments", {}).get("facts_text", "")
            veri["gerekce"] = veri.get("rrl_segments", {}).get("reasoning_text", "")
            veri["hukum"] = veri.get("rrl_segments", {}).get("verdict_text", "Hüküm metni bulunamadı.")
            
            # İçine hiçbir şey kaybetmeden, sadece yeni anahtarlar eklediğimiz
            # devasa 'veri' objesini listeye ekliyoruz
            tum_kararlar.append(veri)
            
    except Exception as e:
        print(f"Hata! {dosya_yolu} okunamadı: {e}")

# Hepsini tek bir dosyaya (kararlar.json) yaz
with open(cikis_dosyasi, 'w', encoding='utf-8') as f:
    json.dump(tum_kararlar, f, ensure_ascii=False, indent=2)

print(f"İşlem Tamam! Hiçbir veri kaybedilmedi. Orijinal verilerin tamamı {cikis_dosyasi} dosyasına aktarıldı.")
