from sklearn.model_selection import train_test_split


class TextMiningManager:
    def __init__(self):
        pass

    def train_test_split(self, data):
        training_data, test_data = train_test_split(data, test_size = 0.3)
        return training_data, test_data