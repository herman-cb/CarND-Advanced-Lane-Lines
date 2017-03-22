import numpy as np

class Tracker:
    def __init__(self, window_w, window_h, margin, ym=1, xm=1, smooth_factor=15):
        self.window_w = window_w
        self.window_h = window_h
        self.margin = margin
        self.ym_per_pixel = ym
        self.xm_per_pixel = xm
        self.smooth_factor = smooth_factor
        self.recent_centers = []

    def find_window_centroids(self, warped):
        window_w = self.window_w
        window_h = self.window_h
        margin = self.margin

        window_centroids = []
        window = np.ones(window_w)

        l_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, :int(warped.shape[1] / 2)], axis=0)
        l_center = np.argmax(np.convolve(window, l_sum)) - window_w / 2
        r_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, int(warped.shape[1] / 2):], axis=0)
        r_center = np.argmax(np.convolve(window, r_sum)) - window_w / 2 + int(warped.shape[1] / 2)

        window_centroids.append((l_center, r_center))

        for level in range(1, int(warped.shape[0] / window_h)):
            image_layer = np.sum(
                warped[int(warped.shape[0] - (level + 1) * window_h):int(warped.shape[0] - level * window_h), :],
                axis=0)
            conv_signal = np.convolve(window, image_layer)
            offset = window_w / 2
            l_min_index = int(max(l_center + offset - margin, 0))
            l_max_index = int(min(l_center + offset + margin, warped.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset

            r_min_index = int(max(r_center + offset - margin, 0))
            r_max_index = int(min(r_center + offset + margin, warped.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset

            window_centroids.append((l_center, r_center))
        self.recent_centers.append(window_centroids)

        return np.average(self.recent_centers[-self.smooth_factor:], axis=0)