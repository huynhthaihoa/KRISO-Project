from matplotlib import pyplot as plt

from network import get_Model
from utils import img_w, img_h, batch_size, img_w, img_h, downsample_factor, max_text_len
from DataGeneration import DataGenerator
from functions import decode_label

model = get_Model(training = False)
model.load_weights('CuDNNLSTM+BN5--29--0.211--0.319.hdf5')


# inp, out = valid_datagen.__getitem__(0)
# images = inp['the_input']
# predictions = model.predict(images[0].reshape(1, img_w, img_h, 1), verbose = 1)
# pred_texts = decode_label(predictions)
# plt.imshow(images[0].reshape(img_w, img_h).T)
# plt.title(pred_texts)