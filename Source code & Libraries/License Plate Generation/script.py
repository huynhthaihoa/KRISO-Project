from PIL import Image
import os
import argparse

def restructureImage(input, xPieces, yPieces):
    '''Restructure the image by dividing the images into (xPieces * yPieces) pieces and merge again into (yPieces, xPieces) image:
    @input: input image path,
    @xPieces: number of pieces per row,
    @yPieces: number of pieces per column
    '''
    im = Image.open(input)
    W, H = im.size
    h = H // yPieces
    w = W // xPieces
    imgMerge = Image.new(im.mode, (w * yPieces, h * xPieces))
    for i in range(0, yPieces):
        for j in range(0, xPieces):
            box = (j * w, i * h, (j + 1) * w, (i + 1) * h)
            a = im.crop(box)
            try:
                imgMerge.paste(a, (i * w, j * h))
            except:
                print("Error occured!")
                pass
    return imgMerge

def changeColor(input, rOld, gOld, bOld, rNew, gNew, bNew, colorKeep=False):
    '''Change pixel color:
    @input: input image path,
    @rOld: r-value of the old color,
    @gOld: g-Value of the old color,
    @bOld: b-Value of the old color,
    @rNew: r-Value of the new color,
    @gNew: g-Value of the new color,
    @bNew: b-Value of the new color,
    @colorKeep: decide the replace color mode (False: replace pixels have the old color with the new color, True: replace pixels don't have the old color with the new color)
    '''
    im = Image.open(input)
    W, H = im.size
    colorOld = (rOld, gOld, bOld)
    colorNew = (rNew, gNew, bNew)
    for i in range(W):
        for j in range(H):
            colorCur = im.getpixel((i, j))
            if colorCur == colorOld:
                print("Ping!: ({0}, {1})".format(i, j))
            if (colorCur == colorOld and colorKeep is False):# or (colorCur != colorOld and colorKeep is True):
                im.putpixel((i, j), colorNew)
            elif(colorKeep is True):
                r, g, b = colorCur
                if(r > rOld or g > gOld or b > bOld):
                    im.putpixel((i, j), colorNew)
    return im

  

    # imgwidth, imgheight = im.size
    # height = imgheight // yPieces
    # width = imgwidth // xPieces
    # imgMerge = Image.new(im.mode, (width * yPieces, height * xPieces))
    # y = 0
    # for i in range(0, yPieces):
    #     for j in range(0, xPieces):
    #         box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
    #         a = im.crop(box)
    #         try:
    #             imgMerge.paste(a, (0, y))
    #             y += imgheight
    #         except:
    #             pass

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input image directory", type=str, default="./in")
parser.add_argument("-o", "--output", help="Output image directory", type=str, default="./out")
parser.add_argument("-r", "--restructImg", help="Restructure the image (True or False)", type=bool, default=False)
parser.add_argument("-x", "--xPieces", help="Number of pieces per row", type=int, default=1)
parser.add_argument("-y", "--yPieces", help="Number of pieces per column", type=int, default=1)
parser.add_argument("-c", "--colorChange", help="Change the pixel color (True or False)", type=bool, default=False)
args = parser.parse_args()

input_dir = args.input
input_dir += '/'
imgs = os.listdir(input_dir)

out_dir = args.output
if os.path.isdir(out_dir) is False:
    os.mkdir(out_dir)
out_dir += '/'

for img in imgs:
    print("Processing image ", img)
    if args.restructImg is True:
        imgMerge = restructureImage(input_dir + img, args.xPieces, args.yPieces)
    if args.colorChange is True:
        imgMerge = changeColor(input_dir + img, 100, 100, 100, 255, 255, 255, True)
    imgMerge.save(out_dir + img)
    print("Processed image ", img)

print("Completed!")
