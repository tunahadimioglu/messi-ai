# 🎬 Duygusal Film Öneri AI'ı

Bu proje, kullanıcıların duygusal deneyimlerini öğrenen ve bu bilgileri kullanarak kişiselleştirilmiş film önerileri yapan bir yapay zeka uygulamasıdır.

## 🌟 Özellikler

- **Duygu Öğrenme**: AI, sizin tanımladığınız duygusal deneyimlerinizi öğrenir
- **Film Analizi**: Film yorumlarındaki duyguları analiz eder ve kategorize eder
- **Kişiselleştirilmiş Öneriler**: Hissetmek istediğiniz duyguya göre film önerir
- **AI Test Sistemi**: AI'ın ne kadar iyi öğrendiğini test edebilirsiniz
- **Veri Kalıcılığı**: Öğrenilen veriler otomatik olarak kaydedilir

## 🎭 Desteklenen Duygular

- **Mutluluk**: Sevinç, neşe, coşku
- **Üzüntü**: Hüzün, melankoli, acı
- **Korku**: Endişe, dehşet, gerilim
- **Öfke**: Sinir, kızgınlık, hiddet
- **Aşk**: Romantizm, sevgi, tutkulu bağlılık
- **Heyecan**: Macera, aksiyon, gerilim
- **Nostalji**: Geçmişe özlem, hatırlama
- **Umut**: İyimserlik, gelecek beklentisi
- **Azgınlık**: Tutku, arzu, cinsel çekicilik
- **Pişmanlık**: Nedamet, vicdan azabı
- **Yetersizlik**: Kendini eksik hissetme
- **Utanç**: Mahcubiyet, sıkılganlık
- **Kaygı**: Endişe, stres, tedirginlik
- **Nefret**: Düşmanlık, tiksinti
- **Kıskançlık**: Çekememezlik, imrenme
- **Bağlanma**: Sevgi, yakınlık, güven

## 🚀 Kurulum

### Gereksinimler
- Python 3.8+
- pip (Python paket yöneticisi)

### Adım 1: Projeyi İndirin
```bash
git clone https://github.com/tunahadimioglu/messi-ai.git
cd messi-ai
```

### Adım 2: Sanal Ortam Oluşturun
```bash
# Windows için
python -m venv myenv
myenv\Scripts\activate

# macOS/Linux için
python3 -m venv myenv
source myenv/bin/activate
```

### Adım 3: Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

## 💻 Kullanım

### İlk Çalıştırma
```bash
python3 main.py
```

### AI'ı Eğitme Süreci
1. **Duygu Öğrenme**: Her duygu için kişisel deneyimlerinizi anlatın
2. **Film Veri Girişi**: Film isimlerini ve yorumlarını girin
3. **Test**: AI'ın ne kadar iyi öğrendiğini test edin

### Ana Menü Seçenekleri
- **1**: Film önerisi al
- **2**: Yeni film verisi ekle
- **3**: AI'ı test et
- **4**: Çıkış

## 📁 Dosya Yapısı

```
duygusal-film-ai/
├── main.py                    # Ana uygulama dosyası
├── requirements.txt           # Python bağımlılıkları
├── README.md                 # Bu dosya
├── emotion_ai_model.pkl      # Eğitilmiş AI modeli (otomatik oluşur)
├── user_emotion_data.json    # Kullanıcı duygu verileri (otomatik oluşur)
├── movie_data.json          # Film verileri (otomatik oluşur)
└── myenv/                   # Sanal ortam klasörü
```

## 🧠 AI Nasıl Çalışır?

1. **Öğrenme**: Sizin verdiğiniz duygusal deneyimlerden kelime kalıpları öğrenir
2. **Analiz**: TF-IDF ve Machine Learning ile metinlerdeki duyguları tespit eder
3. **Eşleştirme**: Film yorumlarını duygusal kategorilere ayırır
4. **Öneri**: İstediğiniz duyguya en uygun filmi önerir

## 🔧 Geliştirme

### Yeni Duygu Ekleme
`main.py` dosyasındaki `emotion_tags` sözlüğüne yeni duygular ekleyebilirsiniz:

```python
emotion_tags = {
    "yeni_duygu": {
        "description": "Açıklama",
        "examples": ["örnek1", "örnek2"]
    }
}
```

### Model Sıfırlama
Eğer AI'ı sıfırlamak istiyorsanız:
```bash
rm emotion_ai_model.pkl user_emotion_data.json movie_data.json
```
