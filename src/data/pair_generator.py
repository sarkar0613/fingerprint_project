import random
from itertools import combinations

class TrainingPairGenerator:
    def __init__(self, labels_tensor, impostor_pair_ratio=1.0):
        self.labels_tensor = labels_tensor
        self.person_dict = self.group_by_prefix()
        self.impostor_pair_ratio = impostor_pair_ratio

    def group_by_prefix(self):
        person_dict = {}
        for i, label in enumerate(self.labels_tensor):
            prefix = label[0].item()
            if prefix not in person_dict:
                person_dict[prefix] = []
            person_dict[prefix].append(i)
        return person_dict

    def generate_genuine_pairs(self):
        genuine_pairs = []
        for prefix, indices in self.person_dict.items():
            if len(indices) > 1:
                for pair in combinations(indices, 2):
                    genuine_pairs.append(pair)
        return genuine_pairs

    def generate_impostor_pairs(self, num_pairs):
        impostor_pairs = []
        all_people = list(self.person_dict.keys())
        for _ in range(num_pairs):
            person1, person2 = random.sample(all_people, 2)
            idx1 = random.choice(self.person_dict[person1])
            idx2 = random.choice(self.person_dict[person2])
            impostor_pairs.append((idx1, idx2))
        return impostor_pairs

    def generate_training_pairs(self):
        genuine_pairs = self.generate_genuine_pairs()
        num_impostor_pairs = int(len(genuine_pairs) * self.impostor_pair_ratio)
        impostor_pairs = self.generate_impostor_pairs(num_impostor_pairs)
        pairs_train = genuine_pairs + impostor_pairs
        labels_train = [1] * len(genuine_pairs) + [0] * len(impostor_pairs)
        return pairs_train, labels_train

    def split_data(self, pairs, labels, train_ratio=0.8):
        genuine_pairs = [pair for pair, label in zip(pairs, labels) if label == 1]
        impostor_pairs = [pair for pair, label in zip(pairs, labels) if label == 0]

        train_genuine_size = int(len(genuine_pairs) * train_ratio)
        train_impostor_size = int(len(impostor_pairs) * train_ratio)

        train_genuine_pairs = genuine_pairs[:train_genuine_size]
        train_impostor_pairs = impostor_pairs[:train_impostor_size]
        test_genuine_pairs = genuine_pairs[train_genuine_size:]
        test_impostor_pairs = impostor_pairs[train_impostor_size:]

        pairs_train = train_genuine_pairs + train_impostor_pairs
        labels_train = [1] * len(train_genuine_pairs) + [0] * len(train_impostor_pairs)
        pairs_test = test_genuine_pairs + test_impostor_pairs
        labels_test = [1] * len(test_genuine_pairs) + [0] * len(test_impostor_pairs)

        return pairs_train, labels_train, pairs_test, labels_test
