from sentence_transformers import SentenceTransformer
import faiss
import openai

# Ön ayarlar
openai.api_key = "YOUR_OPENAI_API_KEY"
model = SentenceTransformer("all-MiniLM-L6-v2")

# Verileri yükle
his_df = pd.read_csv("his_deneyimleri.csv")  # Kolonlar: Duygu, Deneyim
yorum_df = pd.read_csv("film_yorumlari.csv")  # Kolonlar: Film, Yorum

# 1. Tüm deneyim embeddinglerini al
his_embeddings = model.encode(his_df['Deneyim'].tolist(), show_progress_bar=True)

# 2. Tüm yorum embeddinglerini al
yorum_embeddings = model.encode(yorum_df['Yorum'].tolist(), show_progress_bar=True)

# 3. FAISS index kur (yorumlar için)
dim = yorum_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(yorum_embeddings))

# 4. Kullanıcı hissini işleyip benzer yorumları bul

def get_uygun_yorumlar(his_girdileri, top_k=5):
    his_vektoru = model.encode(his_girdileri, show_progress_bar=False)
    if isinstance(his_vektoru[0], list) or isinstance(his_vektoru[0], np.ndarray):
        ortalama_his = np.mean(his_vektoru, axis=0)
    else:
        ortalama_his = his_vektoru
    mesafe, indexler = index.search(np.array([ortalama_his]), top_k)
    yorumlar = yorum_df.iloc[indexler[0]]
    return yorumlar

# 5. GPT'den öneri al

def gpt_ile_oneri_al(his_adi, deneyimler, yorumlar_df):
    yorumlar_text = "\n\n".join([
        f"{row['Film']}: {row['Yorum']}" for _, row in yorumlar_df.iterrows()
    ])
    prompt = f"""
    Kullanıcı aşağıdaki duyguları hissetmek istiyor: {his_adi}

    Bu duygular aşağıdaki deneyimlerle tanımlandı:
    ---
    {chr(10).join(deneyimler)}
    ---

    Bu deneyimlere yakın film yorumları:
    ===
    {yorumlar_text}
    ===

    Yukarıdaki bilgiler doğrultusunda, bu duyguları yaşatabilecek 2 film öner. Kısa açıklama ver.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response['choices'][0]['message']['content']

# 6. Simülasyon: Kullanıcı birden fazla hisle girdi
istekler = [
    "filmden sonra kendime gelemeyeyim",
    "melankoli hissetmek istiyorum"
]

# Deneyim veri setinden bu hislere en yakın deneyimleri bul
uygun_deneyimler = []
uygun_etiketler = []
for istek in istekler:
    his_vec = model.encode(istek)
    sim = model.encode(his_df['Deneyim'].tolist())
    skorlar = np.dot(sim, his_vec) / (np.linalg.norm(sim, axis=1) * np.linalg.norm(his_vec))
    idx = np.argmax(skorlar)
    uygun_deneyimler.append(his_df.iloc[idx]['Deneyim'])
    uygun_etiketler.append(his_df.iloc[idx]['Duygu'])

# Yakın yorumları bul ve GPT'ye gönder
yorumlar = get_uygun_yorumlar(uygun_deneyimler, top_k=6)
oneri = gpt_ile_oneri_al(", ".join(uygun_etiketler), uygun_deneyimler, yorumlar)

print("\n\n🎬 Önerilen Filmler:\n")
print(oneri)
