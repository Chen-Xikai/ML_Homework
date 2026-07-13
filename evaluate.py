import os
import random
import time
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from collections import defaultdict

from model import SiameseContrastive, ContrastiveLoss
from dataset import OmniglotDataset, get_test_alphabets, get_random_writer_split
from utils import set_seed, get_device, format_time


class FewShotEvaluator:
    def __init__(self, model, device, root_dir='./data', seed=42):
        self.model = model
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((105, 105)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.test_dataset = OmniglotDataset(root_dir, split='evaluation')
        self.test_alphabets = get_test_alphabets(root_dir)
        _, _, self.test_writers = get_random_writer_split(seed)
        self.test_writers = set(self.test_writers)
        self.alphabet_to_chars = defaultdict(list)
        self.char_to_images = {}
        self._organize_test_data()
        # 预加载测试集图像到内存
        self.image_cache = {}
        self._warmup_cache()

    def _warmup_cache(self):
        all_paths = set()
        for images in self.char_to_images.values():
            for img_path in images:
                all_paths.add(img_path)
        print(f"  Caching {len(all_paths)} test images...")
        for img_path in all_paths:
            img = Image.open(img_path).convert('L')
            self.image_cache[img_path] = self.transform(img)
        print(f"  Test cache done!")

    def _organize_test_data(self):
        for char_id, images in self.test_dataset.char_to_images.items():
            alphabet = char_id.split('/')[0]
            if alphabet in self.test_alphabets:
                filtered_images = []
                for img_path in images:
                    filename = os.path.basename(img_path)
                    writer_id = int(filename.split('_')[-1].split('.')[0])
                    if writer_id in self.test_writers:
                        filtered_images.append(img_path)
                if len(filtered_images) >= 2:
                    self.alphabet_to_chars[alphabet].append(char_id)
                    self.char_to_images[char_id] = filtered_images
        print(f"Test alphabets: {len(self.alphabet_to_chars)}")

    def extract_embedding(self, image_path):
        self.model.eval()
        with torch.no_grad():
            # 从缓存读取图像
            img = self.image_cache[image_path].unsqueeze(0).to(self.device)
            embedding = self.model.forward_once(img)
            return embedding.cpu().numpy().flatten()

    def create_episode(self, n_way=20):
        available_alphabets = [a for a in self.test_alphabets
                              if len(self.alphabet_to_chars[a]) >= n_way]
        selected_alphabet = np.random.choice(available_alphabets)
        available_chars = self.alphabet_to_chars[selected_alphabet]
        selected_chars = np.random.choice(available_chars, n_way, replace=False)

        writer_ids = list(self.test_writers)
        random.shuffle(writer_ids)
        query_writer = writer_ids[0]
        support_writer = writer_ids[1]

        support_set = []
        for label, char_id in enumerate(selected_chars):
            images = self.char_to_images[char_id]
            support_images = [img for img in images
                             if int(os.path.basename(img).split('_')[-1].split('.')[0]) == support_writer]
            if support_images:
                support_set.append((support_images[0], label))

        query_char_idx = np.random.randint(0, n_way)
        query_char_id = selected_chars[query_char_idx]
        query_images = self.char_to_images[query_char_id]
        query_writer_images = [img for img in query_images
                              if int(os.path.basename(img).split('_')[-1].split('.')[0]) == query_writer]

        query_set = [(query_writer_images[0], query_char_idx)]
        return support_set, query_set, list(range(n_way))

    def run_evaluation(self, n_episodes=400, n_way=20):
        print(f"\n20-way 1-shot evaluation ({n_episodes} episodes)...")
        total_correct = 0
        total_queries = 0
        start_time = time.time()

        for episode_idx in range(n_episodes):
            try:
                support_set, query_set, _ = self.create_episode(n_way=n_way)
                support_embeddings = []
                support_labels = []
                for img_path, label in support_set:
                    embedding = self.extract_embedding(img_path)
                    support_embeddings.append(embedding)
                    support_labels.append(label)
                support_embeddings = np.array(support_embeddings)
                support_labels = np.array(support_labels)

                for img_path, query_label in query_set:
                    query_emb = self.extract_embedding(img_path)
                    distances = [np.linalg.norm(query_emb - se) for se in support_embeddings]
                    predicted_label = support_labels[np.argmin(distances)]
                    total_queries += 1
                    if predicted_label == query_label:
                        total_correct += 1

                if (episode_idx + 1) % 100 == 0:
                    current_acc = 100.0 * total_correct / total_queries
                    print(f"  Episode {episode_idx + 1}/{n_episodes}, Acc: {current_acc:.2f}%")
            except Exception as e:
                continue

        final_accuracy = 100.0 * total_correct / total_queries if total_queries > 0 else 0
        elapsed_time = time.time() - start_time
        print(f"\nResult: {final_accuracy:.2f}% ({total_correct}/{total_queries})")
        print(f"Time: {format_time(elapsed_time)}")
        return {'accuracy': final_accuracy, 'total_correct': total_correct,
                'total_queries': total_queries, 'elapsed_time': elapsed_time}
