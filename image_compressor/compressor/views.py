import os
import numpy as np
import matplotlib.pyplot as plt
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from utils.clustering import kMeans_init_centroids, run_kMeans, find_closest_centroids
from utils.file import format_file_size

def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)
        file_url = fs.url(file_path)

        # Process the image
        image_path = fs.path(file_path)
        original_img = plt.imread(image_path)

        # Check the shape of the image
        # Convert grayscale images to RGB
        if original_img.ndim == 2:  # Grayscale image
            original_img = np.stack((original_img,)*3, axis=-1)  # Convert to RGB
        elif original_img.shape[2] == 4:  # RGBA image
            original_img = original_img[:, :, :3]  # Remove the alpha channel

        # Check if the image is between 0 and 1
        if original_img.max() > 1.0:
            original_img = original_img / 255.0  # Normalize to [0, 1]

        # Reshape the image
        X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))
        K = 16
        max_iters = 10
        initial_centroids = kMeans_init_centroids(X_img, K)
        centroids, idx = run_kMeans(X_img, initial_centroids, max_iters)
        idx = find_closest_centroids(X_img, centroids)
        X_recovered = centroids[idx, :]
        X_recovered = np.reshape(X_recovered, original_img.shape)

        X_recovered = np.clip(X_recovered, 0, 1)

        # Save the compressed image
        folder_path = 'media/compressed_images'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        compressed_file_name = f'compressed.png'
        compressed_image_path = os.path.join(folder_path, compressed_file_name)
        plt.imsave(compressed_image_path, X_recovered)

        compressed_image_url = fs.url(f'compressed_images/{compressed_file_name}')

        return render(request, 'compressor/result.html', {
            'original_image_url': file_url,
            'original_image_size': format_file_size(os.path.getsize(image_path)),
            'compressed_image_url': compressed_image_url,
            'compressed_image_size': format_file_size(os.path.getsize(compressed_image_path))
        })

    return render(request, 'compressor/upload.html')
