import cv2
import numpy as np
from collections import defaultdict
from pythreshold.utils import apply_threshold
from pythreshold.local_th import wolf_threshold

class HoughCircleDetecor:
    def __init__(self, r_range=[150, 300], no_thetas=40, no_circles=40, pixel_treshold=80, treshold=.15,
                 background_type="desk"):

        #iniciate all necessary variables
        self.circle_cand = None
        self.final_circles = []
        self.acc_sorted = {int}
        self.shrinken_circles = []

        self.r_range = r_range
        self.no_thetas = no_thetas
        self.treshold = treshold
        self.no_circles = no_circles

        self.thetas_lst = np.linspace(0, 359, no_thetas).astype(int)
        self.r_lst = np.linspace(r_range[0], r_range[1], no_circles).astype(int)

        self.cos_thetas = np.cos(np.deg2rad(self.thetas_lst))
        self.sin_thetas = np.sin(np.deg2rad(self.thetas_lst))

        self.pixel_treshold = pixel_treshold
        self.background_type = background_type

    def create_circles(self):
        #function that creates circle candidates
        self.circle_cand = [(r, int(r * self.cos_thetas[t]), int(r * self.sin_thetas[t])) for r in self.r_lst for t in
                            range(len(self.thetas_lst))]

    def vote(self):
        """
        This function assign votes by trafersing image. When white pixel occures apply equation for
        previosly prepared circles

        :return: list of circles with assigned no votes
        """
        accumulator = defaultdict(int)

        #iterate over pixels on image
        for y in range(self.img_height):
            print(y)
            for x in range(self.img_width):
                #if white pixel
                if self.e_image[y][x] != 0:
                    #Find circles which corses that circle and give for each a vote
                    for r, rcos, rsin in self.circle_cand:
                        y_cen = int(y - rsin)
                        x_cen = int(x - rcos)
                        try:
                            accumulator[(x_cen, y_cen, r)] += 1
                        except KeyError:
                            accumulator[(x_cen, y_cen, r)] = 1
                        finally:
                            pass
        #sort dict by key value
        self.acc_sorted = sorted(accumulator.items(), key=lambda i: -i[1])

    def find_importand_circles(self):
        """
        This function rejects some circles which are

        :return: list of circle beyond treshold
        """
        for circle, votes in self.acc_sorted:
            if (votes / self.no_circles) > self.treshold:
                self.final_circles.append((circle, votes + np.random.uniform(0, .3, 1)))

    def shrinken_circle_lst(self):
        """
        Function compares circles and chose one from simmilar circles

        :return:
        """
        for idx, circle_a in enumerate(self.final_circles):
            best_circle = True
            circle, v = circle_a
            x, y, r = circle
            for circle_comp, Vcomp in self.final_circles:
                xc, yc, rc = circle_comp
                vc = Vcomp
                if (x == xc) and (y == yc) and (r == rc) and (v == vc):
                    continue
                # compare circle by pixel_treshold
                if (abs(x - xc) < self.pixel_treshold) and (abs(y - yc) < self.pixel_treshold) and (
                        abs(r - rc) < self.pixel_treshold):
                    if v > vc:
                        continue
                        # if v <= vc:
                        #     best_circle = None
                    else:
                        best_circle = None
            if best_circle:
                self.shrinken_circles.append((x, y, r))

    def preprocess_photo(self, input_img):
        """
        Funtion preprocess image appling tresholding method in regard to selected background type

        :param input_img: and image read by cv2
        :return:
        """
        assert self.background_type in ['desk', 'floor'], "background shoud be 'desk' or 'floor'"

        if self.background_type == 'desk':
            image_grey = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(image_grey, (5, 5), cv2.BORDER_DEFAULT)
            self.e_image = cv2.Canny(blur, 100, 200)

        elif self.background_type == "floor":
            image_grey = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(image_grey, (9, 9), cv2.BORDER_DEFAULT)
            wolf_tresholds = wolf_threshold(blur)
            wolf_image = apply_threshold(blur, wolf_tresholds)
            self.e_image = cv2.bitwise_not(wolf_image)

    def __call__(self, PHOTO_PATH):
        self.circle_cand = None
        self.final_circles = []
        self.acc_sorted = {int}
        self.shrinken_circles = []

        input_img = cv2.imread(PHOTO_PATH)
        self.preprocess_photo(input_img)


        self.img_height = self.e_image.shape[0];
        self.img_width = self.e_image.shape[1]

        self.create_circles()
        self.vote()
        self.find_importand_circles()
        self.shrinken_circle_lst()

        return self.shrinken_circles


def show_image(image):
    resized = cv2.resize(image, [640, 640], interpolation=cv2.INTER_AREA)
    cv2.imshow("Image with circles", resized)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    img_path = "data_table/IMG_20230102_012753_jpg.rf.2b27538a1cc33866979b2ea9ada18b9b.jpg"
    input_img = cv2.imread(img_path)

    dict_lst = []

    #function parameters for floor

    # houghCircle = HoughCircleDetecor(r_range=[15, 100], no_thetas=40,\
    #                                  no_circles=20, treshold=.69, pixel_treshold=25, background_type='floor')

    #function parameters for desk
    houghCircle = HoughCircleDetecor(r_range=[150, 300], no_thetas=40, \
                                     no_circles=40, treshold=.15, pixel_treshold=80,
                                     background_type='desk')
    final_list = houghCircle(img_path)

    #example of usage
    for circle in final_list:
        x, y, r = circle
        output_img = cv2.circle(input_img, (x, y), r, (0, 255, 0), 2)

    show_image(output_img)
    del output_img
