import os, random
import cv2, argparse
import numpy as np

# def random_bright(img):
#     '''
#     Set random brightness on image:
#     @img [in]: original image,
#     @img [out]: processed image with random brightness
#     '''
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#     img = np.array(img, dtype=np.float64)
#     random_bright = .5 + np.random.uniform()
#     img[:, :, 2] = img[:, :, 2] * random_bright
#     img[:, :, 2][img[:, :, 2] > 255] = 255
#     img = np.array(img, dtype=np.uint8)
#     img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
#     return img

def random_transform(img, ang_range=6, shear_range=3, trans_range=3):
    '''
    Augment image dataset using perspective, rotation, translation, and shear techniques on an input image:
    @img [in]: original image,
    @ang_range [in]: range of rotation angle,
    @shear_range [in]: range of shear value,
    @trans_range [in]: range of translation value,
    @img [out]: processed image
    '''
    seed = random.randint(0, 2)

    if seed == 1 or seed == 2:
        
        # Rotation
        ang_rot = np.random.uniform(ang_range) - ang_range / 2
        rows, cols, _ = img.shape
        Rot_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 0.9)

        # Translation
        tr_x = trans_range * np.random.uniform() - trans_range / 2
        tr_y = trans_range * np.random.uniform() - trans_range / 2
        Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

        # Shear
        pts1 = np.float32([[5, 5], [20, 5], [5, 20]])
        pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
        pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2
        pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
        shear_M = cv2.getAffineTransform(pts1, pts2)

        img = cv2.warpAffine(img, Rot_M, (cols, rows))
        img = cv2.warpAffine(img, Trans_M, (cols, rows))
        img = cv2.warpAffine(img, shear_M, (cols, rows))
    
    if seed == 0 or seed == 2:
        # Perspective
        w, h, _ = img.shape
        pts1 = np.float32([[0, 0], [0, w], [h, 0], [h, w]])
        begin, end = 0, 45
        pts2 = np.float32([[random.randint(begin, end), random.randint(begin, end)],
                       [random.randint(begin, end), w - random.randint(begin, end)],
                       [h - random.randint(begin, end), random.randint(begin, end)],
                       [h - random.randint(begin, end), w - random.randint(begin, end)]])
        M = cv2.getPerspectiveTransform(pts1, pts2)

        img = cv2.warpPerspective(img, M, (h, w))

    return img

def random_bright(img):
    '''
    Augment image dataset by adding random blur and brightness on input image:
    @img [in]: original image,
    @img [out]: processed image
    '''    
    # Brightness
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = np.array(img, dtype=np.float64)
    random_bright = .4 + np.random.uniform()
    img[:, :, 2] = img[:, :, 2] * random_bright
    img[:, :, 2][img[:, :, 2] > 255] = 255
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    # Blur
    blur_value = random.randint(0,5) * 2 + 1
    img = cv2.blur(img, (blur_value, blur_value))

    return img

class ImageGenerator:
    def __init__(self, save_path):
        '''
        Intialize image generator:
        @save_path [in]: directory to save generated images
        '''
        self.save_path = save_path
        # Plate
        self.plate = cv2.imread("plate_y.jpg")
        # self.plate2 = cv2.imread("plate_y.jpg")
        # self.plate3 = cv2.imread("plate_g.jpg")

        # loading Region
        file_path = "region_y/"
        file_list = os.listdir(file_path)
        self.Region = list()
        self.region_list = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Region.append(img)
            self.region_list.append(file[0:-4])

        # loading Region vertical
        file_path = "region_y_vertical/"
        file_list = os.listdir(file_path)
        self.Region_vert = list()
        self.region_list_vert = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Region_vert.append(img)
            self.region_list_vert.append(file[0:-4])

        # loading Number
        file_path = "num_y/"
        file_list = os.listdir(file_path)
        self.Number = list()
        self.number_list = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Number.append(img)
            self.number_list.append(file[0:-4])

        # loading Char
        file_path = "char_truck/"
        file_list = os.listdir(file_path)
        self.char_list = list()
        self.Char = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Char.append(img)
            self.char_list.append(file[0:-4])

    def Type_1(self, num, save=True, augment=False):
        '''
        Generate truck license plate type 1 (1 line):
        @num [in]: number of generated images,
        @save [in]: save (True) or show (False) generated images,
        @augment [in]: augment (True) or keep originally (False) generated image 
        '''
        region = [cv2.resize(region, (48, 95)) for region in self.Region_vert]
        number = [cv2.resize(number, (56, 83)) for number in self.Number]
        char = [cv2.resize(char1, (60, 83)) for char1 in self.Char]
        Plate = cv2.resize(self.plate, (562, 123))

        for i, _ in enumerate(range(num)):
            Plate = cv2.resize(self.plate, (562, 123))
            label = ""
            # row -> y , col -> x
            row, col = 13, 24  # row + 83, col + 56

            #region
            label += self.region_list_vert[i % 16]
            Plate[row : row + 95, col : col + 48, :] = region[i % 16]
            col += 60

            # number 1
            rand_int = random.randint(8, 9)
            label += self.number_list[rand_int]
            Plate[row + 6 : row + 89, col : col + 56, :] = number[rand_int]
            col += 56

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row + 6 : row + 89, col : col + 56, :] = number[rand_int]
            col += 56

            # character
            rand_int = random.randint(0, 3)
            label += self.char_list[rand_int]
            Plate[row + 6 : row + 89, col : col + 60, :] = char[rand_int]
            col += (60 + 36)

            # number 3
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row + 6 : row + 89, col : col + 56, :] = number[rand_int]
            col += 56

            # number 4
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row + 6 : row + 89, col : col + 56, :] = number[rand_int]
            col += 56

            # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row + 6 : row + 89, col : col + 56, :] = number[rand_int]
            col += 56

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row + 6 : row + 89, col : col + 56, :] = number[rand_int]
            col += 56
            if augment is True:
                Plate = random_transform(Plate)
            Plate = random_bright(Plate)
            if save:
                while os.path.exists(self.save_path + label + ".jpg"):
                    label += '_'
                cv2.imwrite(self.save_path + label + ".jpg", Plate)
            else:
                cv2.imshow(label, Plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def Type_2(self, num, save=True, augment=False):
        '''
        Generate truck license plate type 2 (1 line):
        @num [in]: number of generated images,
        @save [in]: save (True) or show (False) generated images,
        @augment [in]: augment (True) or keep originally (False) generated image 
        '''
        region = [cv2.resize(region, (49, 83)) for region in self.Region_vert]
        number = [cv2.resize(number, (45, 83)) for number in self.Number]
        char = [cv2.resize(char1, (49, 70)) for char1 in self.Char]
        Plate = cv2.resize(self.plate, (392, 155))

        for i in range(num):
            Plate = cv2.resize(self.plate, (392, 155))
            label = ""
            row, col = 46, 10  # row + 83, col + 56
            
            #region
            label += self.region_list_vert[i % 16]
            Plate[row:row + 83, col:col + 49, :] = region[i % 16]
            col += 49 + 2
            
            # number 1
            rand_int = random.randint(8, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 45, :] = number[rand_int]
            col += 45

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 45, :] = number[rand_int]
            col += 45

            # char
            rand_int = random.randint(0, 3)
            label += self.char_list[rand_int]
            Plate[row + 12:row + 82, col + 2:col + 49 + 2, :] = char[rand_int]
            col += 49 + 2

            # number 3
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col + 2:col + 45 + 2, :] = number[rand_int]
            col += 45 + 2

            # number 4
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 45, :] = number[rand_int]
            col += 45

            # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 45, :] = number[rand_int]
            col += 45

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 45, :] = number[rand_int]
            col += 45
            if augment is True:
                Plate = random_transform(Plate)#random_bright(Plate)
            Plate = random_bright(Plate)
            if save:
                while os.path.exists(self.save_path + label + ".jpg"):
                    label += '_'
                cv2.imwrite(self.save_path + label + ".jpg", Plate)
            else:
                cv2.imshow(label, Plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def Type_3(self, num, save=True, augment=False):
        '''
        Generate truck license plate type 3 (2 lines):
        @num [in]: number of generated images,
        @save [in]: save (True) or show (False) generated images,
        @augment [in]: augment (True) or keep originally (False) generated image 
        '''
        number1 = [cv2.resize(number, (44, 60)) for number in self.Number]
        number2 = [cv2.resize(number, (64, 90)) for number in self.Number]
        region = [cv2.resize(region, (88, 60)) for region in self.Region]
        char = [cv2.resize(char1, (64, 90)) for char1 in self.Char]

        for i, _ in enumerate(range(num)):
            Plate = cv2.resize(self.plate, (336, 170))

            label = str()
            # row -> y , col -> x
            row, col = 8, 76

            # region
            label += self.region_list[i % 16]
            Plate[row:row + 60, col:col + 88, :] = region[i % 16]
            col += 88 + 8

            # number 1
            rand_int = random.randint(8, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 60, col:col + 44, :] = number1[rand_int]
            col += 44

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 60, col:col + 44, :] = number1[rand_int]

            row, col = 72, 8

            # character
            rand_int = random.randint(0, 3)
            label += self.char_list[rand_int]
            Plate[row:row + 90, col:col + 64, :] = char[rand_int]
            col += 64

            # number 3
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 90, col:col + 64, :] = number2[rand_int]
            col += 64

            # number 4
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 90, col:col + 64, :] = number2[rand_int]
            col += 64

            # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 90, col:col + 64, :] = number2[rand_int]
            col += 64

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 90, col:col + 64, :] = number2[rand_int]
            if augment is True:
                Plate = random_transform(Plate)#random_bright(Plate)
            Plate = random_bright(Plate)
            if save:
                while os.path.exists(self.save_path + label + ".jpg"):
                    label += '_'
                cv2.imwrite(self.save_path + label + ".jpg", Plate)
            else:
                cv2.imshow(label, Plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--img_dir", help="Directory to save images",
                    type=str, default="DB/")
parser.add_argument("-n", "--num", help="Number of images",
                    type=int, default=100)
parser.add_argument("-s", "--save", help="Save or show images",
                    type=bool, default=True)
parser.add_argument("-a", "--augment", help="Augment images or not",
                    type=bool, default=False)
args = parser.parse_args()


img_dir = args.img_dir
if os.path.isdir(img_dir) is False:
    os.mkdir(img_dir)
img_dir += '/'
A = ImageGenerator(img_dir)

num_img = args.num
num_original = num_img // 3
num_augmented = num_img - num_original
Save = args.save
Augment = args.augment

A.Type_1(num_augmented, Save, True)
print("Type 1 (augmented) finish")
A.Type_1(num_original, Save)
print("Type 1 (original) finish")
# A.Type_2(num_img, Save)
# print("Type 2 finish")
A.Type_3(num_augmented, Save, True)
print("Type 3 (augmented) finish")
A.Type_3(num_original, Save)
print("Type 3 (original) finish")
# A.Type_4(num_img, save=Save)
# print("Type 4 finish")
# A.Type_5(num_img, save=Save)
# print("Type 5 finish")
