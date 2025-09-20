import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, X):
        np.random.seed(42)
        random_indices = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iters):
            labels = self._assign_clusters(X)
            new_centroids = []

            for i in range(self.k):
                cluster_points = X[labels == i]
                if len(cluster_points) == 0:
                    new_centroids.append(X[np.random.randint(0, len(X))])
                else:
                    new_centroids.append(cluster_points.mean(axis=0))

            new_centroids = np.array(new_centroids)

            if np.all(np.linalg.norm(new_centroids - self.centroids, axis=1) < self.tol):
                break

            self.centroids = new_centroids

        self.labels = labels

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def predict(self, X):
        return self._assign_clusters(X)


if __name__ == "__main__":
    caminho_img = input("Digite o caminho da imagem: ").strip()
    k = int(input("Digite o valor de K (número de clusters): "))

    img = cv2.imread(caminho_img)
    if img is None:
        raise FileNotFoundError("Imagem não encontrada! Verifique o caminho.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    escolha = input("Deseja redimensionar a imagem para acelerar? (s/n): ").strip().lower()
    if escolha == "s":
        largura = int(input("Digite a largura desejada (ex: 200): "))
        altura = int(input("Digite a altura desejada (ex: 200): "))
        img_proc = cv2.resize(img, (largura, altura))
    else:
        img_proc = img.copy()

    pixels = img_proc.reshape(-1, 3)

    kmeans = KMeans(k=k)
    kmeans.fit(pixels)

    segmented_pixels = kmeans.centroids[kmeans.labels].astype(np.uint8)
    segmented_img = segmented_pixels.reshape(img_proc.shape)

    unique, counts = np.unique(kmeans.labels, return_counts=True)
    freq = counts / counts.sum()
    sorted_idx = np.argsort(freq)[::-1]  

    palette = np.zeros((50, 300, 3), dtype=np.uint8)
    start = 0
    for idx in sorted_idx:
        color = kmeans.centroids[idx].astype(np.uint8)
        end = start + int(freq[idx] * palette.shape[1])
        palette[:, start:end] = color
        start = end

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img_proc)
    axs[0].set_title("Imagem Original")
    axs[0].axis("off")

    axs[1].imshow(segmented_img)
    axs[1].set_title(f"Segmentada (K={k})")
    axs[1].axis("off")

    axs[2].imshow(palette)
    axs[2].set_title("Paleta de cores")
    axs[2].axis("off")

    plt.show()

    nome_arquivo = os.path.splitext(os.path.basename(caminho_img))[0]
    saida_img = f"{nome_arquivo}_segmentada_K{k}.png"
    saida_palette = f"{nome_arquivo}_palette_K{k}.png"

    cv2.imwrite(saida_img, cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(saida_palette, cv2.cvtColor(palette, cv2.COLOR_RGB2BGR))

    print(f"✅ Imagem segmentada salva como: {saida_img}")
    print(f"✅ Paleta de cores salva como: {saida_palette}")
