import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from skimage import color

# Função para ler uma imagem
def ler_imagem(caminho):
    return cv2.imread(caminho)

# Função para exibir uma imagem
def exibir_imagem(imagem, titulo="Imagem"):
    # Verificar se a imagem é colorida (3 canais) ou em tons de cinza (1 canal)
    if len(imagem.shape) == 3:
        # Converter imagem colorida para RGB para exibição no Matplotlib
        imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        plt.imshow(imagem_rgb)
    else:
        # A imagem é em tons de cinza
        plt.imshow(imagem, cmap='gray')

    plt.title(titulo)
    plt.axis('off')
    plt.show()


# Função para extrair características de uma imagem
def extrair_caracteristicas(imagem):
    altura, largura, _ = imagem.shape
    return altura, largura


# Função para obter o tamanho de um arquivo em KB
def obter_tamanho_em_kb(caminho_arquivo):
    tamanho_em_bytes = os.path.getsize(caminho_arquivo)
    tamanho_em_kb = tamanho_em_bytes / 1024  # Converter bytes para kilobytes
    return tamanho_em_kb


# Função para calcular a quantidade de cores únicas em uma imagem
def contar_cores_unicas(imagem):
    cores_unicas = np.unique(imagem.reshape(-1, imagem.shape[2]), axis=0)
    return len(cores_unicas)


# Função para aplicar o algoritmo k-médias em uma imagem
def aplicar_kmeans(imagem, k):
    altura, largura, _ = imagem.shape
    imagem_reshape = imagem.reshape((altura * largura, 3))

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # Defina explicitamente n_init
    kmeans.fit(imagem_reshape)

    # Atribuir cada pixel ao cluster mais próximo
    rotulos = kmeans.predict(imagem_reshape)

    # Calcular os centroides dos clusters
    centroides = kmeans.cluster_centers_

    # Reconstruir a imagem com base nos centroides
    imagem_segmentada = centroides[rotulos].reshape(imagem.shape).astype(np.uint8)  # Converta para uint8

    return imagem_segmentada, centroides



# Função para calcular a diferença entre duas imagens
def calcular_diferenca(imagem1, imagem2):
    return np.sum(np.abs(imagem1 - imagem2))

# Caminho da imagem
caminho_imagem = "imagem.png"

# Leitura da imagem
imagem_original = ler_imagem(caminho_imagem)

# Exibir imagem original
exibir_imagem(imagem_original, "Imagem Original")

# Extrair características da imagem original
altura_original, largura_original = extrair_caracteristicas(imagem_original)
print(f"Características da imagem original: Altura={altura_original}, Largura={largura_original}")

# Obter o tamanho da imagem original em KB
tamanho_original_kb = obter_tamanho_em_kb(caminho_imagem)
print(f"Tamanho da imagem original: {tamanho_original_kb:.2f} KB")

# Calcular a quantidade de cores únicas na imagem original
cores_unicas_original = contar_cores_unicas(imagem_original)
print(f"Quantidade de cores únicas na imagem original: {cores_unicas_original}")

# Definir valores de k para experimentação
valores_k = [110,40,20,15,9,4,2]

# Aplicar o algoritmo k-médias para diferentes valores de k
for k in valores_k:
    imagem_segmentada, centroides = aplicar_kmeans(imagem_original, k)

    # Exibir imagem segmentada
    exibir_imagem(imagem_segmentada, f"Imagem Segmentada (k={k})")

    # Extrair características da imagem segmentada
    altura_segmentada, largura_segmentada = extrair_caracteristicas(imagem_segmentada)
    print(f"Características da imagem segmentada (k={k}): Altura={altura_segmentada}, Largura={largura_segmentada}")

     # Obter o tamanho da imagem segmentada em KB
    tamanho_segmentada_kb = obter_tamanho_em_kb(caminho_imagem_segmentada)
    print(f"Tamanho da imagem segmentada (k={k}): {tamanho_segmentada_kb:.2f} KB")

     # Calcular a quantidade de cores únicas na imagem segmentada
    cores_unicas_segmentada = contar_cores_unicas(imagem_segmentada)
    print(f"Quantidade de cores únicas na imagem segmentada (k={k}): {cores_unicas_segmentada}")


