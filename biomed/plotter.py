from matplotlib import pyplot as plt
import seaborn


class Plotter:
    def plot_target_distribution(self, input_data):
        plt.figure(figsize=(12, 9))
        plt.subplot(211)
        seaborn.countplot(input_data['doid'])
        plt.subplot(212)
        seaborn.countplot(input_data['is_cancer'])
        plt.show()
