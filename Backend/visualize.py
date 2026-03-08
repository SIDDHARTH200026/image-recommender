import matplotlib.pyplot as plt
import numpy as np

def visualize_recommendations(dataset, query_idx, top_indices):
    fig, axes = plt.subplots(1, 6, figsize=(15, 3))
    images = [dataset[query_idx][0]] + [dataset[i][0] for i in top_indices]
    titles = ['QUERY'] + [f'{i}' for i in top_indices]
    
    for i, (img, title) in enumerate(zip(images, titles)):
        img = img.permute(1,2,0).clamp(0,1).numpy()
        axes[i].imshow(img)
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("📊 Saved: results.png")
