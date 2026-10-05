import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import gc

# Sayfa Yapılandırması
st.set_page_config(page_title="Neural Style Transfer", page_icon="🎨", layout="centered")

st.title("🎨 Neural Style Transfer (Klasik Algoritma)")
st.markdown("""
Bu uygulama, Gatys et al. tarafından geliştirilen orijinal **Neural Style Transfer** algoritmasını kullanır. 
Görsellerinizin içeriğini ve stilini $J = \\alpha J_{content} + \\beta J_{style}$ formülü ile birleştirir.
""")

# Cihaz Seçimi
device = torch.device("cpu")

# 1. Görsel Ön İşleme
def image_loader(image_file, max_size=400): # RAM için 400px ideal boyuttur
    image = Image.open(image_file).convert('RGB')
    
    size = max(image.size)
    if size > max_size:
        scale = max_size / float(size)
        new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
        image = image.resize(new_size, Image.LANCZOS)

    loader = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return loader(image).unsqueeze(0).to(device, torch.float)

def deprocess_image(tensor):
    image = tensor.cpu().clone().detach().squeeze(0)
    unnormalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    image = unnormalize(image)
    image = torch.clamp(image, 0, 1)
    return transforms.ToPILImage()(image)

# 2. Gram Matrisi Hesaplama
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    G = torch.mm(features, features.t())
    return G.div(c * h * w)

# 3. Model Yükleme
@st.cache_resource
def load_vgg_model():
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
    selected_layers = {
        '0': 'conv1_1', 
        '5': 'conv2_1', 
        '10': 'conv3_1', 
        '19': 'conv4_1', 
        '28': 'conv5_1', 
        '21': 'conv4_2'
    }
    model = nn.Sequential()
    for name, layer in vgg._modules.items():
        model.add_module(name, layer)
    for param in model.parameters():
        param.requires_grad = False
    return model.to(device).eval(), selected_layers

model, selected_layers = load_vgg_model()

def get_features(x, model, layers):
    features = {}
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# 4. Arayüz
col1, col2 = st.columns(2)
with col1:
    content_file = st.file_uploader("İçerik Görüntüsü (Content)", type=["jpg", "png", "jpeg"])
with col2:
    style_file = st.file_uploader("Stil Görüntüsü (Style)", type=["jpg", "png", "jpeg"])

st.subheader("⚙️️ Hiperparametreler")
alpha = st.slider("İçerik Ağırlığı (Alpha)", 1.0, 10.0, 1.0, step=0.5)
beta = st.slider("Stil Ağırlığı (Beta)", 1e3, 1e6, 1e5, step=1e4, format="%e")
steps = st.slider("Optimizasyon Adımı (Steps)", 30, 150, 80, step=10) # 80 adım CPU RAM için en güvenli aralıktır

if st.button("🚀 Stil Transferini Başlat"):
    if content_file and style_file:
        # Bellek Temizliği
        gc.collect()

        content_tensor = image_loader(content_file)
        style_tensor = image_loader(style_file)

        target_tensor = content_tensor.clone().requires_grad_(True)

        content_features = get_features(content_tensor, model, selected_layers)
        style_features = get_features(style_tensor, model, selected_layers)

        style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features if layer != 'conv4_2'}
        style_weights = {'conv1_1': 1.0, 'conv2_1': 0.8, 'conv3_1': 0.5, 'conv4_1': 0.3, 'conv5_1': 0.2}

        optimizer = optim.LBFGS([target_tensor])

        st.info("ℹ️ İşlem yürütülüyor. Sayfayı kapatmadan canlı ilerlemeyi takip edebilirsiniz.")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        preview_image = st.empty()

        run = [0]
        while run[0] <= steps:
            def closure():
                optimizer.zero_grad()
                target_features = get_features(target_tensor, model, selected_layers)

                content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

                style_loss = 0
                for layer in style_weights:
                    target_gram = gram_matrix(target_features[layer])
                    style_gram = style_grams[layer]
                    style_loss += style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)

                total_loss = (alpha * content_loss) + (beta * style_loss)
                total_loss.backward()

                run[0] += 1
                
                # Güncelleme
                progress = min(run[0] / steps, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"⏳ Adım: {run[0]}/{steps} | Toplam Maliyet (J): {total_loss.item():.2f}")

                if run[0] % 10 == 0 or run[0] == steps:
                    preview_image.image(deprocess_image(target_tensor), caption=f"Canlı Önizleme (Adım {run[0]})", use_column_width=True)

                return total_loss

            optimizer.step(closure)

        st.success("🎉 Stil Transferi Tamamlandı!")
        st.image(deprocess_image(target_tensor), caption="Üretilen Görsel", use_column_width=True)
        gc.collect()
    else:
        st.warning("Lütfen hem içerik hem de stil görseli yükleyin.")
