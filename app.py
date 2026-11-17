import gc
import io

import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageOps
from torchvision import models, transforms

# Sayfa Yapılandırması
st.set_page_config(page_title="Neural Style Transfer", page_icon="🎨", layout="centered")

st.title("🎨 Neural Style Transfer (Klasik Algoritma)")
st.markdown("""
Bu uygulama, Gatys et al. tarafından geliştirilen orijinal **Neural Style Transfer** algoritmasını kullanır. 
Görsellerinizin içeriğini ve stilini $J = \\alpha J_{content} + \\beta J_{style}$ formülü ile birleştirir.
""")

# Cihaz Seçimi: GPU varsa kullan, yoksa CPU (Streamlit Cloud'da CPU olacaktır)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# VGG19 katman indeksleri (ReLU çıktıları). Model 29. katmandan sonra kesilir.
LAYERS = {
    "1": "relu1_1",
    "6": "relu2_1",
    "11": "relu3_1",
    "20": "relu4_1",
    "29": "relu5_1",
    "22": "relu4_2",  # içerik katmanı
}
LAST_LAYER = 29
CONTENT_LAYER = "relu4_2"
STYLE_WEIGHTS = {
    "relu1_1": 1.0,
    "relu2_1": 0.8,
    "relu3_1": 0.5,
    "relu4_1": 0.3,
    "relu5_1": 0.2,
}


# 1. Görsel Ön İşleme
def image_loader(image_file, max_size=400):  # RAM için 400px ideal boyuttur
    image = Image.open(image_file)
    image = ImageOps.exif_transpose(image)  # telefon fotoğraflarının yönünü düzelt
    image = image.convert("RGB")

    size = max(image.size)
    if size > max_size:
        scale = max_size / float(size)
        new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
        image = image.resize(new_size, Image.LANCZOS)

    loader = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    return loader(image).unsqueeze(0).to(device, torch.float)


def deprocess_image(tensor):
    image = tensor.cpu().clone().detach().squeeze(0)
    unnormalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(MEAN, STD)],
        std=[1 / s for s in STD],
    )
    image = unnormalize(image)
    image = torch.clamp(image, 0, 1)
    return transforms.ToPILImage()(image)


# 2. Gram Matrisi Hesaplama
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.reshape(c, h * w)
    G = torch.mm(features, features.t())
    return G.div(c * h * w)


# 3. Model Yükleme (yalnızca ihtiyaç duyulan katmanlara kadar)
@st.cache_resource
def load_vgg_model():
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
    model = nn.Sequential()
    for name, layer in vgg.named_children():
        if int(name) > LAST_LAYER:
            break
        if isinstance(layer, nn.ReLU):
            layer = nn.ReLU(inplace=False)  # in-place ReLU, kaydedilen özellikleri bozabilir
        model.add_module(name, layer)
    for param in model.parameters():
        param.requires_grad = False
    return model.to(device).eval()


model = load_vgg_model()


def get_features(x, model, layers):
    features = {}
    for name, layer in model.named_children():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features


# 4. Stil Transferi
def run_style_transfer(content_file, style_file, alpha, beta, steps,
                       progress_bar, status_text, preview_image):
    content_tensor = image_loader(content_file)
    style_tensor = image_loader(style_file)

    with torch.no_grad():
        content_features = get_features(content_tensor, model, LAYERS)
        style_features = get_features(style_tensor, model, LAYERS)
        style_grams = {layer: gram_matrix(style_features[layer]) for layer in STYLE_WEIGHTS}
        content_target = content_features[CONTENT_LAYER]

    target_tensor = content_tensor.clone().requires_grad_(True)

    # LBFGS tüm adım bütçesini kendisi yönetir; closure yaklaşık `steps` kez çağrılır
    optimizer = optim.LBFGS([target_tensor], max_iter=steps)

    run = [0]

    def closure():
        optimizer.zero_grad()
        target_features = get_features(target_tensor, model, LAYERS)

        content_loss = torch.mean((target_features[CONTENT_LAYER] - content_target) ** 2)

        style_loss = 0
        for layer, weight in STYLE_WEIGHTS.items():
            target_gram = gram_matrix(target_features[layer])
            style_loss = style_loss + weight * torch.mean((target_gram - style_grams[layer]) ** 2)

        total_loss = (alpha * content_loss) + (beta * style_loss)
        total_loss.backward()

        run[0] += 1
        progress_bar.progress(min(run[0] / steps, 1.0))
        status_text.text(f"⏳ Adım: {min(run[0], steps)}/{steps} | Toplam Maliyet (J): {total_loss.item():.2f}")

        if run[0] % 10 == 0:
            preview_image.image(
                deprocess_image(target_tensor),
                caption=f"Canlı Önizleme (Adım {run[0]})",
            )

        return total_loss

    optimizer.step(closure)
    progress_bar.progress(1.0)

    result = deprocess_image(target_tensor)

    # Bellek temizliği
    del content_tensor, style_tensor, target_tensor, content_features, style_features, style_grams
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def image_to_png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# 5. Arayüz
col1, col2 = st.columns(2)
with col1:
    content_file = st.file_uploader("İçerik Görüntüsü (Content)", type=["jpg", "png", "jpeg"])
with col2:
    style_file = st.file_uploader("Stil Görüntüsü (Style)", type=["jpg", "png", "jpeg"])

st.subheader("⚙️ Hiperparametreler")
alpha = st.slider("İçerik Ağırlığı (Alpha)", 1.0, 10.0, 1.0, step=0.5)
beta = st.slider("Stil Ağırlığı (Beta)", 1e3, 1e6, 1e5, step=1e4, format="%e")
steps = st.slider("Optimizasyon Adımı (Steps)", 30, 150, 80, step=10)  # 80 adım CPU RAM için güvenli aralıktır

if st.button("🚀 Stil Transferini Başlat"):
    if content_file and style_file:
        st.session_state.pop("result_png", None)  # eski sonucu temizle

        info = st.info("ℹ️ İşlem yürütülüyor. Sayfayı kapatmadan canlı ilerlemeyi takip edebilirsiniz.")
        progress_bar = st.progress(0)
        status_text = st.empty()
        preview_image = st.empty()

        try:
            result = run_style_transfer(
                content_file, style_file, alpha, beta, steps,
                progress_bar, status_text, preview_image,
            )
            st.session_state["result_png"] = image_to_png_bytes(result)
        except RuntimeError as e:
            st.error(f"İşlem sırasında hata oluştu (muhtemelen bellek yetersiz): {e}")
        finally:
            info.empty()
            progress_bar.empty()
            status_text.empty()
            preview_image.empty()
            gc.collect()
    else:
        st.warning("Lütfen hem içerik hem de stil görseli yükleyin.")

# Sonuç, session_state'te saklanır: slider'a dokunsan da kaybolmaz
if "result_png" in st.session_state:
    st.success("🎉 Stil Transferi Tamamlandı!")
    st.image(st.session_state["result_png"], caption="Üretilen Görsel")
    st.download_button(
        label="⬇️ Görseli İndir (PNG)",
        data=st.session_state["result_png"],
        file_name="stylized.png",
        mime="image/png",
    )