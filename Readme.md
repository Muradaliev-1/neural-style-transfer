# 🎨 Neural Style Transfer

Bir fotoğrafın içeriğini, başka bir görselin sanatsal stiliyle birleştiren web uygulaması. Gatys et al. tarafından geliştirilen klasik **Neural Style Transfer** algoritmasını kullanır ve **Streamlit** ile hazırlanmıştır.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://neural-style-muradaliev.streamlit.app)

### 👉 [Canlı demoyu dene](https://neural-style-muradaliev.streamlit.app)

> **Not:** Uygulama ücretsiz bir CPU sunucusunda çalışır. Bir süre kullanılmazsa uyku moduna geçebilir; "Wake up" butonuna basıp birkaç saniye beklemen gerekebilir. İşlem birkaç dakika sürebilir.

---

## Nasıl çalışır?

1. **İçerik görseli** ve **stil görseli** yüklenir.
2. Önceden eğitilmiş **VGG19** ağı, iki görselden farklı katmanlarda özellik çıkarır.
3. Stil, katman çıktılarının **Gram matrisleri** ile temsil edilir.
4. Başlangıçta içerik görselinin kopyası olan bir görsel, **L-BFGS** optimizasyonu ile adım adım güncellenir.

Minimize edilen toplam maliyet:

```
J = α · J_content + β · J_style
```

## Özellikler

- Canlı ilerleme çubuğu ve her 10 adımda bir önizleme
- Ayarlanabilir içerik ağırlığı (α), stil ağırlığı (β) ve adım sayısı
- Sonucu PNG olarak indirme
- Telefon fotoğraflarının yönünü otomatik düzeltme (EXIF)
- GPU varsa otomatik kullanım, yoksa CPU

## Parametre ipuçları

| Parametre | Etkisi |
|-----------|--------|
| **Alpha (içerik ağırlığı)** | Arttıkça sonuç orijinal fotoğrafa daha sadık kalır |
| **Beta (stil ağırlığı)** | Arttıkça stil daha baskın olur |
| **Steps (adım)** | Arttıkça sonuç daha oturur ama işlem uzar |

Görseller bellek tasarrufu için en uzun kenarı **400 px** olacak şekilde küçültülür.

## Yerelde çalıştırma

```bash
git clone https://github.com/Muradaliev-1/neural-style-transfer.git
cd neural-style-transfer

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

İlk çalıştırmada VGG19 ağırlıkları (~550 MB) otomatik indirilir.

`requirements.txt` CPU sürümü torch kurar. NVIDIA GPU kullanmak istiyorsan, torch'u [pytorch.org](https://pytorch.org/get-started/locally/) üzerindeki talimatla kur.

## Kullanılan teknolojiler

- [Python](https://www.python.org/)
- [PyTorch](https://pytorch.org/) ve torchvision (VGG19)
- [Streamlit](https://streamlit.io/)
- [Pillow](https://python-pillow.org/)

## Kaynak

L. A. Gatys, A. S. Ecker, M. Bethge — *A Neural Algorithm of Artistic Style* (2015). [arXiv:1508.06576](https://arxiv.org/abs/1508.06576)