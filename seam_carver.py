import matplotlib.pyplot as plt
import cv2 as cv
import json
import numpy as np
import matplotlib
matplotlib.use("TkAgg")


class SeamCarver:
    def __init__(self, input_img, specifications) -> None:
        self.input_img = input_img
        self.specifications = specifications
        self.reqd_h = specifications["height"]
        self.reqd_w = specifications["width"]
        self.output_img = input_img.copy()
        self.img_h, self.img_w = input_img.shape[:2]

    def validate(self):
        print(f"Input Dimensions: {self.img_w} x {self.img_h}")
        print(f"Required Dimensions: {self.reqd_w} x {self.reqd_h}")
        return self.reqd_h <= self.img_h and self.reqd_w <= self.img_w

    def seam_carving_vertical(self, img, target_width):
        """Carve vertical seams from a given image array."""
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.float64)
        num_seams = img.shape[1] - target_width
        carved_img = img.copy()

        for i in range(num_seams):
            print(f"Pass {i+1}/{num_seams}")
            H, W = img_gray.shape
            gx = cv.Sobel(img_gray, cv.CV_64F, 1, 0, ksize=3)
            gy = cv.Sobel(img_gray, cv.CV_64F, 0, 1, ksize=3)
            energy = (np.abs(gx) + np.abs(gy)).astype(np.float64)

            # DP table
            dp = energy.copy()
            for row in range(1, H):
                for col in range(W):
                    if col == 0:
                        dp[row, col] += min(dp[row-1, col], dp[row-1, col+1])
                    elif col == W-1:
                        dp[row, col] += min(dp[row-1, col-1], dp[row-1, col])
                    else:
                        dp[row, col] += min(dp[row-1, col-1], dp[row-1, col], dp[row-1, col+1])

            # Backtrack seam
            seam = np.zeros(H, dtype=np.int32)
            seam[-1] = np.argmin(dp[-1])
            for row in range(H-2, -1, -1):
                prev = seam[row+1]
                if prev == 0:
                    choices = [dp[row, prev], dp[row, prev+1]]
                    seam[row] = prev + np.argmin(choices)
                elif prev == W-1:
                    choices = [dp[row, prev-1], dp[row, prev]]
                    seam[row] = prev - 1 + np.argmin(choices)
                else:
                    choices = [dp[row, prev-1], dp[row, prev], dp[row, prev+1]]
                    seam[row] = prev - 1 + np.argmin(choices)

            # Remove seam
            img_gray = np.array([np.delete(img_gray[row], seam[row]) for row in range(H)])
            carved_img = np.array([np.delete(carved_img[row], seam[row], axis=0) for row in range(H)])

        return carved_img

    def seam_carving(self):
        self.output_img = self.seam_carving_vertical(self.output_img, self.reqd_w)
        rotated = np.rot90(self.output_img, k=-1)
        carved_rotated = self.seam_carving_vertical(rotated, self.reqd_h)
        self.output_img = np.rot90(carved_rotated, k=1)

    def show_images(self):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(cv.cvtColor(self.input_img, cv.COLOR_BGR2RGB))
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(cv.cvtColor(self.output_img, cv.COLOR_BGR2RGB))
        plt.title("Seam Carved")
        plt.axis("off")
        plt.show()

    def save_output(self, path="output.png"):
        cv.imwrite(path, self.output_img)


def main():
    input_img = cv.imread("data/human.png")
    with open("data/specifications.json") as f:
        specifications = json.load(f)

    carver = SeamCarver(input_img, specifications)
    if not carver.validate():
        print("Required dimensions are larger than image dimensions.")
        return

    # Vertical carving
    carver.seam_carving()

    # Save FIRST, then show (in case visualization crashes)
    carver.save_output("output_carved.png")
    print("Saved output_carved.png")
    carver.show_images()


if __name__ == '__main__':
    main()
