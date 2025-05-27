import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
import json
from datetime import datetime
import pickle

# Geliştirilmiş duygu etiketleri - sizin tanımladığınız kategoriler
emotion_tags = {
    'mutluluk': {
        'experiences': [],
        'keywords': ['sevinç', 'neşe', 'coşku', 'keyif', 'mutlu'],
        'description': 'Sevinç, neşe ve pozitif enerji'
    },
    'hüzün': {
        'experiences': [],
        'keywords': ['üzüntü', 'keder', 'melankoli', 'acı', 'hüzün'],
        'description': 'Üzüntü, keder ve melankolik duygular'
    },
    'heyecan': {
        'experiences': [],
        'keywords': ['heyecan', 'coşku', 'enerji', 'dinamizm', 'aksiyon'],
        'description': 'Yüksek enerji, heyecan ve dinamizm'
    },
    'korku': {
        'experiences': [],
        'keywords': ['korku', 'endişe', 'kaygı', 'gerilim', 'tedirginlik'],
        'description': 'Korku, endişe ve gerilim'
    },
    'nostalji': {
        'experiences': [],
        'keywords': ['nostalji', 'geçmiş', 'hatıra', 'özlem', 'anı'],
        'description': 'Geçmişe özlem ve nostaljik duygular'
    },
    'romantizm': {
        'experiences': [],
        'keywords': ['aşk', 'romantik', 'sevgi', 'tutku', 'romantizm'],
        'description': 'Aşk, romantizm ve duygusal bağ'
    },
    'azgınlık': {
        'experiences': [],
        'keywords': ['azgınlık', 'tutku', 'arzu', 'şehvet', 'istek', 'çekicilik'],
        'description': 'Güçlü arzu, tutku ve çekicilik'
    },
    'pişmanlık': {
        'experiences': [],
        'keywords': ['pişmanlık', 'nedamet', 'vicdan azabı', 'keşke', 'üzülme'],
        'description': 'Geçmiş kararlar için duyulan pişmanlık ve vicdan azabı'
    },
    'yetersizlik': {
        'experiences': [],
        'keywords': ['yetersizlik', 'eksiklik', 'beceriksizlik', 'güvensizlik', 'başarısızlık'],
        'description': 'Kendini yeterli görmeme, eksiklik hissi'
    },
    'utanç': {
        'experiences': [],
        'keywords': ['utanç', 'mahcubiyet', 'sıkılma', 'rezillik', 'utanma'],
        'description': 'Utanma, mahcubiyet ve sıkılma duyguları'
    },
    'kaygı': {
        'experiences': [],
        'keywords': ['kaygı', 'endişe', 'stres', 'gerginlik', 'tedirginlik'],
        'description': 'Gelecek hakkında endişe, kaygı ve stres'
    },
    'nefret': {
        'experiences': [],
        'keywords': ['nefret', 'kin', 'öfke', 'tiksinti', 'iğrenme'],
        'description': 'Güçlü olumsuz duygular, nefret ve tiksinti'
    },
    'kıskançlık': {
        'experiences': [],
        'keywords': ['kıskançlık', 'çekememezlik', 'imrenme', 'haset', 'rekabet'],
        'description': 'Başkalarını kıskanma, haset ve çekememezlik'
    },
    'bağlanma': {
        'experiences': [],
        'keywords': ['bağlanma', 'sevgi', 'yakınlık', 'bağlılık', 'sadakat'],
        'description': 'Güçlü duygusal bağ, bağlılık ve yakınlık'
    }
}

# Film-duygu veritabanı (geliştirilmiş)
movie_emotions = pd.DataFrame(columns=[
    'movie_name', 'review', 'predicted_emotion', 'confidence', 
    'user_id', 'timestamp', 'user_confirmed_emotion'
])

class AITestSystem:
    def __init__(self, recommender):
        self.recommender = recommender
        
    def test_emotion_prediction(self):
        """AI'ın duygu tahmin yeteneğini test eder"""
        print("\n🧪 AI DUYGU TAHMİN TESTİ")
        print("=" * 40)
        
        # Test cümleleri - her duygu için açık örnekler
        test_cases = {
            'mutluluk': [
                "Bu film beni çok mutlu etti, sürekli gülümsedim",
                "Harika bir komedi, kahkaha attım",
                "Çok eğlenceli ve neşeli bir filmdi"
            ],
            'hüzün': [
                "Bu film beni çok üzdü, ağladım",
                "Melankolik ve hüzünlü bir hikaye",
                "Çok duygusal, gözyaşlarımı tutamadım"
            ],
            'heyecan': [
                "Nefes kesen aksiyon sahneleri vardı",
                "Çok heyecanlı, kenarında oturdum",
                "Adrenalin dolu bir macera filmi"
            ],
            'korku': [
                "Çok korkunçtu, gece uyuyamadım",
                "Gerilim dolu, sürekli irkiliyordum",
                "Korkudan gözlerimi kapadım"
            ],
            'nostalji': [
                "Bu film beni çocukluğuma götürdü",
                "Geçmişi hatırlatan nostaljik bir film",
                "Eski günleri özlettiren bir hikaye"
            ],
            'romantizm': [
                "Çok romantik bir aşk hikayesi",
                "Kalp ısıtan romantik sahneler",
                "Aşkın gücünü anlatan güzel bir film"
            ],
            'azgınlık': [
                "Çok tutkulu ve ateşli sahneler vardı",
                "Güçlü bir çekicilik ve arzu hissettim",
                "Sıcak ve arzulu bir atmosfer yaratmış"
            ],
            'pişmanlık': [
                "Karakterin yaptığı hatalar için çok pişman oldum",
                "Keşke farklı seçimler yapsaydı diye düşündüm",
                "Vicdan azabı çeken karakterle empati kurdum"
            ],
            'yetersizlik': [
                "Kendimi çok yetersiz hissettim",
                "Karakterin başarısızlığı beni etkiledi",
                "Kendi eksikliklerimi düşündüm"
            ],
            'utanç': [
                "Karakterin utanç verici durumu beni mahcup etti",
                "O kadar utandım ki yüzümü kapattım",
                "Rezil olan karakter için sıkıldım"
            ],
            'kaygı': [
                "Sürekli endişeli ve gergin hissettim",
                "Ne olacağı konusunda çok kaygılandım",
                "Stresli sahneler beni tedirgin etti"
            ],
            'nefret': [
                "Kötü karaktere karşı nefret duydum",
                "O kadar tiksindim ki izleyemedim",
                "İğrenç sahneler vardı"
            ],
            'kıskançlık': [
                "Karakterin başarısını kıskandım",
                "Onun yerine ben olmak isterdim",
                "Haset duyguları uyandırdı"
            ],
            'bağlanma': [
                "Karakterlere çok bağlandım",
                "Güçlü bir duygusal bağ kurdum",
                "Onları sevdim ve yakın hissettim"
            ]
        }
        
        total_tests = 0
        correct_predictions = 0
        detailed_results = []
        
        for expected_emotion, test_sentences in test_cases.items():
            print(f"\n🎯 {expected_emotion.upper()} testi:")
            
            for sentence in test_sentences:
                if self.recommender.is_trained:
                    predicted_emotion, confidence, _ = self.recommender.predict_emotion_with_confidence(sentence)
                    
                    is_correct = predicted_emotion == expected_emotion
                    total_tests += 1
                    if is_correct:
                        correct_predictions += 1
                    
                    # Sonucu göster
                    status = "✅" if is_correct else "❌"
                    print(f"  {status} '{sentence[:50]}...'")
                    print(f"     Beklenen: {expected_emotion} | Tahmin: {predicted_emotion} | Güven: {confidence:.2f}")
                    
                    detailed_results.append({
                        'sentence': sentence,
                        'expected': expected_emotion,
                        'predicted': predicted_emotion,
                        'confidence': confidence,
                        'correct': is_correct
                    })
                else:
                    print("  ⚠️  AI henüz eğitilmedi!")
                    return
        
        # Genel sonuçlar
        if total_tests > 0:
            accuracy = (correct_predictions / total_tests) * 100
            print(f"\n📊 GENEL SONUÇLAR:")
            print(f"✅ Doğru tahmin: {correct_predictions}/{total_tests}")
            print(f"📈 Doğruluk oranı: {accuracy:.1f}%")
            
            # Duygu bazında analiz
            emotion_accuracy = {}
            for emotion in test_cases.keys():
                emotion_results = [r for r in detailed_results if r['expected'] == emotion]
                emotion_correct = sum(1 for r in emotion_results if r['correct'])
                emotion_total = len(emotion_results)
                emotion_accuracy[emotion] = (emotion_correct / emotion_total * 100) if emotion_total > 0 else 0
            
            print(f"\n🎭 DUYGU BAZINDA BAŞARI:")
            for emotion, acc in emotion_accuracy.items():
                print(f"  {emotion}: {acc:.1f}%")
            
            # Öneriler
            if accuracy < 50:
                print(f"\n💡 ÖNERİLER:")
                print("- Daha fazla duygu örneği ekleyin")
                print("- Daha detaylı deneyimler anlatın")
                print("- AI'ı yeniden eğitin")
            elif accuracy < 80:
                print(f"\n💡 İYİ! Daha da geliştirebilirsiniz:")
                print("- Zayıf duygular için daha fazla örnek ekleyin")
            else:
                print(f"\n🎉 MÜKEMMEL! AI çok iyi öğrenmiş!")
        
        return detailed_results
    
    def interactive_test(self):
        """Kullanıcının kendi cümlelerini test etmesini sağlar"""
        print("\n🎮 İNTERAKTİF TEST")
        print("=" * 30)
        print("Kendi cümlelerinizi yazın, AI'ın ne tahmin ettiğini görün!")
        print("'çıkış' yazarak ana menüye dönebilirsiniz.\n")
        
        while True:
            test_sentence = input("Test cümlesi: ").strip()
            
            if test_sentence.lower() in ['çıkış', 'exit', 'quit', '']:
                break
            
            if self.recommender.is_trained:
                predicted_emotion, confidence, all_scores = self.recommender.predict_emotion_with_confidence(test_sentence)
                
                print(f"\n🤖 AI'ın Tahmini:")
                print(f"   🎯 En güçlü duygu: {predicted_emotion} ({confidence:.2f})")
                print(f"   📊 Tüm duygu skorları:")
                
                # Skorları sırala
                sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
                for emotion, score in sorted_scores:
                    bar = "█" * int(score * 20)  # 20 karakterlik bar
                    print(f"      {emotion:12}: {score:.3f} {bar}")
                print()
            else:
                print("⚠️  AI henüz eğitilmedi!")
                break

class AdvancedEmotionalRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
        self.emotion_classifier = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=1000)),
            ('classifier', MultinomialNB())
        ])
        self.is_trained = False
        self.user_feedback_data = []
        
    def collect_emotion_examples(self):
        """Kullanıcılardan duygu örnekleri topla - geliştirilmiş versiyon"""
        print("🎭 Duygu Öğrenme Aşaması")
        print("=" * 50)
        print("AI'ın sizi daha iyi anlaması için, her duyguyu ne zaman hissettiğinizi anlatın.")
        print("Mümkün olduğunca detaylı ve kişisel deneyimlerinizi paylaşın.\n")
        
        for emotion, data in emotion_tags.items():
            print(f"\n🎯 {emotion.upper()} - {data['description']}")
            print(f"Örnek durumlar: {', '.join(data['keywords'])}")
            print(f"\n'{emotion}' duygusunu hayatınızda ne zaman, hangi deneyimlerde hissedersiniz?")
            print("(Birkaç farklı örnek verebilirsiniz, 'tamam' yazarak geçebilirsiniz)")
            
            while True:
                experience = input(f"{emotion} deneyimi: ").strip()
                if experience.lower() in ['tamam', 'geç', 'next', '']:
                    break
                if len(experience) > 10:  # Minimum uzunluk kontrolü
                    emotion_tags[emotion]['experiences'].append(experience)
                    print("✅ Eklendi! Başka bir deneyim eklemek isterseniz yazın, yoksa 'tamam' yazın.")
                else:
                    print("Lütfen daha detaylı bir deneyim anlatın.")
    
    def generate_synthetic_examples(self):
        """Anahtar kelimelerden sentetik örnekler oluştur"""
        templates = [
            "Bu durum beni {keyword} hissettirdi",
            "{keyword} dolu bir deneyim yaşadım",
            "O an {keyword} duygusunu yoğun şekilde hissettim",
            "{keyword} bir anıydı, unutamam",
            "Hayatımda {keyword} hissettiğim nadir anlardan biriydi"
        ]
        
        for emotion, data in emotion_tags.items():
            keywords = data['keywords']
            for keyword in keywords:
                for template in templates[:2]:  # Her anahtar kelime için 2 örnek
                    synthetic_text = template.format(keyword=keyword)
                    emotion_tags[emotion]['experiences'].append(synthetic_text)
    
    def train_emotion_classifier(self):
        """Geliştirilmiş duygu sınıflandırıcıyı eğit"""
        print("\n🤖 AI Eğitim Aşaması")
        print("=" * 30)
        
        # Sentetik örnekler ekle
        self.generate_synthetic_examples()
        
        # Eğitim verisi hazırla
        X = []
        y = []
        
        for emotion, data in emotion_tags.items():
            experiences = data['experiences']
            if len(experiences) == 0:
                print(f"⚠️  {emotion} için örnek bulunamadı, varsayılan örnekler ekleniyor...")
                # Varsayılan örnekler ekle
                default_examples = [f"Bu beni {emotion} hissettirdi", f"{emotion} bir deneyimdi"]
                experiences.extend(default_examples)
                emotion_tags[emotion]['experiences'] = experiences
            
            X.extend(experiences)
            y.extend([emotion] * len(experiences))
        
        if len(X) < 6:  # Minimum veri kontrolü
            print("❌ Yeterli eğitim verisi yok! Lütfen daha fazla örnek ekleyin.")
            return False
        
        # Modeli eğit
        try:
            self.emotion_classifier.fit(X, y)
            self.is_trained = True
            print(f"✅ Model başarıyla eğitildi! ({len(X)} örnek ile)")
            return True
        except Exception as e:
            print(f"❌ Model eğitimi başarısız: {e}")
            return False
    
    def predict_emotion_with_confidence(self, text):
        """Metinden duygu tahmin et - güven skoru ile"""
        if not self.is_trained:
            return "bilinmeyen", 0.0, {}
        
        try:
            prediction = self.emotion_classifier.predict([text])[0]
            probabilities = self.emotion_classifier.predict_proba([text])[0]
            confidence = max(probabilities)
            
            # Tüm duygu skorlarını al
            emotion_scores = dict(zip(self.emotion_classifier.classes_, probabilities))
            
            return prediction, confidence, emotion_scores
        except:
            return "bilinmeyen", 0.0, {}
    
    def add_movie_with_learning(self, name, review, user_id="default_user"):
        """Film ekle ve AI'ın tahminini kullanıcıya sor"""
        global movie_emotions
        
        print(f"\n🎬 Film Ekleniyor: {name}")
        print(f"📝 Yorum: {review}")
        
        # AI'ın tahmini
        predicted_emotion, confidence, emotion_scores = self.predict_emotion_with_confidence(review)
        
        print(f"\n🤖 AI'ın Tahmini: {predicted_emotion} (Güven: {confidence:.2f})")
        
        # Kullanıcıya sor
        print("\n🎯 Bu tahmin doğru mu? Eğer değilse, gerçek duyguyu seçin:")
        emotions_list = list(emotion_tags.keys())
        for i, emotion in enumerate(emotions_list, 1):
            print(f"{i}. {emotion}")
        print(f"{len(emotions_list) + 1}. AI'ın tahmini doğru")
        
        try:
            choice = input("Seçiminiz (sayı): ").strip()
            if choice == str(len(emotions_list) + 1):
                confirmed_emotion = predicted_emotion
                print("✅ AI'ın tahmini onaylandı!")
            else:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(emotions_list):
                    confirmed_emotion = emotions_list[choice_idx]
                    print(f"✅ Gerçek duygu: {confirmed_emotion}")
                else:
                    confirmed_emotion = predicted_emotion
                    print("⚠️  Geçersiz seçim, AI'ın tahmini kullanılıyor.")
        except:
            confirmed_emotion = predicted_emotion
            print("⚠️  Geçersiz giriş, AI'ın tahmini kullanılıyor.")
        
        # Veritabanına ekle
        new_row = pd.DataFrame([{
            'movie_name': name,
            'review': review,
            'predicted_emotion': predicted_emotion,
            'confidence': confidence,
            'user_id': user_id,
            'timestamp': datetime.now(),
            'user_confirmed_emotion': confirmed_emotion
        }])
        
        movie_emotions = pd.concat([movie_emotions, new_row], ignore_index=True)
        
        # Öğrenme için geri bildirim kaydet
        if predicted_emotion != confirmed_emotion:
            self.user_feedback_data.append({
                'review_text': review,
                'predicted': predicted_emotion,
                'actual': confirmed_emotion,
                'confidence': confidence
            })
            print("📚 AI bu geri bildirimden öğrenecek!")
        
        # Belirli aralıklarla yeniden eğit
        if len(self.user_feedback_data) >= 5:
            self.retrain_with_feedback()
    
    def retrain_with_feedback(self):
        """Kullanıcı geri bildirimleriyle modeli yeniden eğit"""
        print("\n🔄 AI öğrenmeye devam ediyor...")
        
        # Mevcut eğitim verisi
        X_original = []
        y_original = []
        for emotion, data in emotion_tags.items():
            X_original.extend(data['experiences'])
            y_original.extend([emotion] * len(data['experiences']))
        
        # Geri bildirim verilerini ekle
        X_feedback = [fb['review_text'] for fb in self.user_feedback_data]
        y_feedback = [fb['actual'] for fb in self.user_feedback_data]
        
        # Birleştir ve yeniden eğit
        X_all = X_original + X_feedback
        y_all = y_original + y_feedback
        
        try:
            self.emotion_classifier.fit(X_all, y_all)
            print("✅ AI güncellendi ve daha akıllı hale geldi!")
            self.user_feedback_data = []  # Geri bildirim listesini temizle
        except Exception as e:
            print(f"⚠️  Güncelleme başarısız: {e}")
    
    def recommend_movie_advanced(self, desired_emotion, user_id="default_user"):
        """Geliştirilmiş film önerisi"""
        print(f"\n🎯 '{desired_emotion}' duygusunu hissetmek istiyorsunuz...")
        
        # Bu duyguya sahip filmleri bul
        matching_movies = movie_emotions[
            movie_emotions['user_confirmed_emotion'] == desired_emotion
        ]
        
        if len(matching_movies) == 0:
            # Alternatif: AI'ın tahmin ettiği duygulara bak
            matching_movies = movie_emotions[
                movie_emotions['predicted_emotion'] == desired_emotion
            ]
        
        if len(matching_movies) == 0:
            return f"😔 Henüz '{desired_emotion}' duygusuna uygun film bulunamadı. Daha fazla film ekleyin!"
        
        # Güven skoruna göre sırala
        matching_movies = matching_movies.sort_values('confidence', ascending=False)
        
        # En iyi öneriler
        top_movies = matching_movies.head(3)
        
        recommendations = []
        for _, movie in top_movies.iterrows():
            recommendations.append({
                'name': movie['movie_name'],
                'confidence': movie['confidence'],
                'review_sample': movie['review'][:100] + "..." if len(movie['review']) > 100 else movie['review']
            })
        
        return recommendations
    
    def save_model(self, filename="emotion_ai_model.pkl"):
        """Modeli kaydet"""
        model_data = {
            'emotion_tags': emotion_tags,
            'classifier': self.emotion_classifier,
            'is_trained': self.is_trained,
            'movie_emotions': movie_emotions
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"💾 Model kaydedildi: {filename}")
    
    def load_model(self, filename="emotion_ai_model.pkl"):
        """Modeli yükle"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            global emotion_tags, movie_emotions
            emotion_tags = model_data['emotion_tags']
            self.emotion_classifier = model_data['classifier']
            self.is_trained = model_data['is_trained']
            movie_emotions = model_data['movie_emotions']
            
            print(f"📂 Model yüklendi: {filename}")
            return True
        except FileNotFoundError:
            print(f"⚠️  Model dosyası bulunamadı: {filename}")
            return False

def reset_ai_completely():
    """AI'ı tamamen sıfırla"""
    global emotion_tags, movie_emotions
    
    print("⚠️  AI'ı tamamen sıfırlamak istediğinizden emin misiniz?")
    print("Bu işlem tüm öğrenilen verileri silecek!")
    confirm = input("'EVET' yazarak onaylayın: ").strip()
    
    if confirm == "EVET":
        # Emotion tags'i sıfırla
        for emotion in emotion_tags:
            emotion_tags[emotion]['experiences'] = []
        
        # Film veritabanını sıfırla
        movie_emotions = pd.DataFrame(columns=[
            'movie_name', 'review', 'predicted_emotion', 'confidence', 
            'user_id', 'timestamp', 'user_confirmed_emotion'
        ])
        
        # Model dosyasını sil
        import os
        if os.path.exists("emotion_ai_model.pkl"):
            os.remove("emotion_ai_model.pkl")
        
        print("🔄 AI tamamen sıfırlandı! Program yeniden başlatılıyor...")
        return True
    else:
        print("❌ Sıfırlama iptal edildi.")
        return False

def main():
    print("🎬 Duygusal Film Öneri AI'ı")
    print("=" * 40)
    
    recommender = AdvancedEmotionalRecommender()
    test_system = AITestSystem(recommender)  # Test sistemi oluştur
    
    # Önceki modeli yüklemeye çalış
    if not recommender.load_model():
        print("\n🆕 Yeni AI eğitimi başlıyor...")
    
    # Duygu örnekleri topla
    recommender.collect_emotion_examples()
    
    # Sınıflandırıcıyı eğit
    if not recommender.train_emotion_classifier():
        print("❌ AI eğitimi başarısız! Program sonlandırılıyor.")
        return
    
    # Modeli kaydet
    recommender.save_model()
    
    print(f"\n✅ AI hazır! Toplam {len(movie_emotions)} film veritabanında.")
    
    while True:
        print("\n" + "="*50)
        print("🎬 MENÜ")
        print("1. 📽️  Film ekle ve AI'ı eğit")
        print("2. 🎯 Film önerisi al")
        print("3. 📊 İstatistikleri görüntüle")
        print("4. 🧪 AI'ı test et")
        print("5. 🎮 İnteraktif test")
        print("6. 💾 Modeli kaydet")
        print("7. 🔄 AI'ı tamamen sıfırla")
        print("8. 🚪 Çıkış")
        
        choice = input("\nSeçiminiz: ").strip()
        
        if choice == "1":
            print("\n📽️  YENİ FİLM EKLEME")
            name = input("Film adı: ").strip()
            if name:
                review = input("Film hakkındaki yorumunuz (detaylı): ").strip()
                if review:
                    recommender.add_movie_with_learning(name, review)
                else:
                    print("⚠️  Yorum boş olamaz!")
            else:
                print("⚠️  Film adı boş olamaz!")
            
        elif choice == "2":
            print("\n🎯 FİLM ÖNERİSİ")
            print("Hangi duyguyu hissetmek istersiniz?")
            emotions_list = list(emotion_tags.keys())
            for i, emotion in enumerate(emotions_list, 1):
                print(f"{i}. {emotion} - {emotion_tags[emotion]['description']}")
            
            try:
                emotion_choice = input("Seçiminiz (sayı veya isim): ").strip().lower()
                
                # Sayı ile seçim
                if emotion_choice.isdigit():
                    choice_idx = int(emotion_choice) - 1
                    if 0 <= choice_idx < len(emotions_list):
                        desired_emotion = emotions_list[choice_idx]
                    else:
                        print("⚠️  Geçersiz seçim!")
                        continue
                # İsim ile seçim
                elif emotion_choice in emotions_list:
                    desired_emotion = emotion_choice
                else:
                    print("⚠️  Geçersiz duygu!")
                    continue
                
                recommendations = recommender.recommend_movie_advanced(desired_emotion)
                
                if isinstance(recommendations, str):
                    print(recommendations)
                else:
                    print(f"\n🎬 '{desired_emotion}' için önerilerim:")
                    for i, rec in enumerate(recommendations, 1):
                        print(f"\n{i}. 🎭 {rec['name']}")
                        print(f"   📊 Güven: {rec['confidence']:.2f}")
                        print(f"   💭 Örnek yorum: {rec['review_sample']}")
                        
            except Exception as e:
                print(f"⚠️  Hata: {e}")
                
        elif choice == "3":
            print("\n📊 İSTATİSTİKLER")
            print(f"📽️  Toplam film sayısı: {len(movie_emotions)}")
            
            if len(movie_emotions) > 0:
                emotion_counts = movie_emotions['user_confirmed_emotion'].value_counts()
                print("\n🎭 Duygu dağılımı:")
                for emotion, count in emotion_counts.items():
                    print(f"   {emotion}: {count} film")
                
                print(f"\n🤖 AI doğruluk oranı: {(movie_emotions['predicted_emotion'] == movie_emotions['user_confirmed_emotion']).mean():.2%}")
            
        elif choice == "4":
            test_system.test_emotion_prediction()
            
        elif choice == "5":
            test_system.interactive_test()
            
        elif choice == "6":
            recommender.save_model()
            
        elif choice == "7":
            if reset_ai_completely():
                break  # Programı yeniden başlat
                
        elif choice == "8":
            print("👋 Görüşmek üzere!")
            break
        
        else:
            print("⚠️  Geçersiz seçim!")

if __name__ == "__main__":
    main()
