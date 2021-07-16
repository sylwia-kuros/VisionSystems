import matplotlib.pyplot as plt
import numpy as np
import cv2

model = cv2.imread('images/wzor.pgm')
model = cv2.cvtColor(model, cv2.COLOR_BGR2GRAY)

img = cv2.imread('images/domek_r0.pgm')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# wzorzec z zerami
model2 = np.zeros(img.shape)
model2[0:model.shape[0], 0:model.shape[1]] = model

# korelacja w dziedzinie częstotliwości
model2_fft = np.fft.fft2(model2)
img_fft = np.fft.fft2(img)

# sprzężenie i moduł
conjugation = np.conj(model2_fft) * img_fft
conjugation = conjugation / np.abs(conjugation)
module = np.abs(np.fft.ifft2(conjugation))

# współrzędne maksimum w obrazie modułu transformaty odwrotnej
y, x = np.unravel_index(np.argmax(module), module.shape)

# macierz translacji
translation_matrix = np.float32([[1, 0, x], [0, 1, y]])

# przekształcenie afiniczne
translated_image = cv2.warpAffine(model, translation_matrix, (img.shape[1], img.shape[0]))

plt.imshow(model, cmap='gray')
plt.title('Model')
plt.axis('off')
plt.show()

plt.imshow(img, cmap='gray')
plt.plot(x, y, '*r')
plt.title('Houses')
plt.axis('off')
plt.show()

plt.imshow(translated_image, cmap='gray')
plt.title('Translated image')
plt.axis('off')
plt.show()

def hanning2D(n):
    h = np.hanning(n)
    return np.sqrt(np.outer(h, h))


def highpassFilter(size):
    rows = np.cos(np.pi*np.array([-0.5 + x / (size[0] - 1) for x in range(size[0])]))
    cols = np.cos(np.pi*np.array([-0.5 + x / (size[1] - 1) for x in range(size[1])]))
    X = np.outer(rows, cols)
    return (1.0 - X) * (2.0 - X)

# wzorzec
model_in = cv2.imread('images/domek_r0_64.pgm')
model_in = cv2.cvtColor(model_in, cv2.COLOR_BGR2GRAY)

# obrazy obrócone
for i in range(0, 330, 30):
    img = 'images/domek_r' + str(i) + '.pgm'
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    model2 = np.zeros(img.shape)
    model = model_in * hanning2D(model_in.shape[0])
    model2[0:model.shape[0], 0:model.shape[1]] = model

    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)

    model2_fft = np.fft.fft2(model2)
    model2_fft = np.fft.fftshift(model2_fft)

    # filtracja
    img_f = np.abs(img_fft) * highpassFilter(img_fft.shape)
    model2_f = np.abs(model2_fft) * highpassFilter(model2_fft.shape)

    img_fft = np.abs(img_f)
    model2_fft = np.abs(model2_f)

    M = model2_fft.shape[0] / np.log(model2_fft.shape[0] // 2)
    center = (model2_fft.shape[0] // 2, model2_fft.shape[1] // 2)

    model2_logpolar = cv2.logPolar(model2_fft, center, M, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
    img_logpolar = cv2.logPolar(img_fft, center, M, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)

    img_fft = np.fft.fft2(img_logpolar)
    model2_fft = np.fft.fft2(model2_logpolar)

    # sprzężenie i moduł
    conjugation = np.conj(model2_fft) * img_fft
    conjugation = conjugation / np.abs(conjugation)
    module = np.abs(np.fft.ifft2(conjugation))

    # współrzędne maksimum w obrazie modułu transformaty odwrotnej
    alph, logp = np.unravel_index(np.argmax(module), module.shape)

    logpolar_size, alpha_size = img_logpolar.shape

    if logp > logpolar_size // 2:
        w = logpolar_size - logp  # powiększenie
    else:
        w = - logp  # pomniejszenie

    scale = np.exp(w / M)  # M jest parametrem funkcji cv2.logPolar
    print('Skala:', scale)

    alpha = (alph * 360.0) / alpha_size
    angle1 = - alpha
    angle2 = 180 - alpha

    im = np.zeros(img.shape)
    x1 = int((img.shape[0] - model.shape[0]) / 2)
    x2 = int((img.shape[0] + model.shape[0]) / 2)

    y1 = int((img.shape[1] - model.shape[1]) / 2)
    y2 = int((img.shape[1] + model.shape[1]) / 2)

    im[x1:x2, y1:y2] = model_in

    center_rotation = (im.shape[0] / 2 - 0.5, im.shape[1] / 2 - 0.5)

    matrix1 = cv2.getRotationMatrix2D(center_rotation, angle1, scale)
    im_rotated1 = cv2.warpAffine(im, matrix1, im.shape)

    matrix2 = cv2.getRotationMatrix2D(center_rotation, angle2, scale)
    im_rotated2 = cv2.warpAffine(im, matrix2, im.shape)

    im_fft_1 = np.fft.fft2(im_rotated1)
    im_fft_2 = np.fft.fft2(im_rotated2)
    img_fft = np.fft.fft2(img)

    conjugation1 = np.conj(im_fft_1) * img_fft
    conjugation1 = conjugation1 / np.abs(conjugation1)
    module_1 = np.abs(np.fft.ifft2(conjugation1))

    conjugation2 = np.conj(im_fft_2) * img_fft
    conjugation2 = conjugation2 / np.abs(conjugation2)
    module_2 = np.abs(np.fft.ifft2(conjugation2))

    if np.amax(module_1) > np.amax(module_2):
        module_m = module_1
        pattern_m = im_rotated1
    else:
        module_m = module_2
        pattern_m = im_rotated2

    y, x = np.unravel_index(np.argmax(module_m), module_m.shape)
    if x > img.shape[0] - 5:
        x = x - img.shape[0]

    print('Wektory przesunięcia x, y -> obrazy obrócone:', x, y)
    matrix = np.float32([[1, 0, x], [0, 1, y]])  # x, y - wektor przesunięcia
    im_rotated = cv2.warpAffine(pattern_m, matrix, (img.shape[1], img.shape[0]))

    plt.figure()
    plt.imshow(im_rotated, cmap='gray')
    plt.axis('off')
    plt.title('Model after processing')
    plt.show()

    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title('Analyzed image')
    plt.show()

# obrazy przesunięte
for i in range(10, 80, 10):
    img = 'images/domek_s' + str(i) + '.pgm'
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    model2 = np.zeros(img.shape)
    model = model_in * hanning2D(model_in.shape[0])
    model2[0:model.shape[0], 0:model.shape[1]] = model

    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)

    model2_fft = np.fft.fft2(model2)
    model2_fft = np.fft.fftshift(model2_fft)

    # filtracja
    img_f = np.abs(img_fft) * highpassFilter(img_fft.shape)
    model2_f = np.abs(model2_fft) * highpassFilter(model2_fft.shape)

    img_fft = np.abs(img_f)
    model2_fft = np.abs(model2_f)

    M = model2_fft.shape[0] / np.log(model2_fft.shape[0] // 2)
    center = (model2_fft.shape[0] // 2, model2_fft.shape[1] // 2)

    model2_logpolar = cv2.logPolar(model2_fft, center, M, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
    img_logpolar = cv2.logPolar(img_fft, center, M, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)

    img_fft = np.fft.fft2(img_logpolar)
    model2_fft = np.fft.fft2(model2_logpolar)

    # sprzężenie i moduł
    conjugation = np.conj(model2_fft) * img_fft
    conjugation = conjugation / np.abs(conjugation)
    module = np.abs(np.fft.ifft2(conjugation))

    # współrzędne maksimum w obrazie modułu transformaty odwrotnej
    alph, logp = np.unravel_index(np.argmax(module), module.shape)

    logpolar_size, alpha_size = img_logpolar.shape

    if logp > logpolar_size // 2:
        w = logpolar_size - logp  # powiększenie
    else:
        w = - logp  # pomniejszenie

    scale = np.exp(w / M)  # M jest parametrem funkcji cv2.logPolar
    print('Skala:', scale)

    alpha = (alph * 360.0) / alpha_size
    angle1 = - alpha
    angle2 = 180 - alpha

    im = np.zeros(img.shape)
    x1 = int((img.shape[0] - model.shape[0]) / 2)
    x2 = int((img.shape[0] + model.shape[0]) / 2)

    y1 = int((img.shape[1] - model.shape[1]) / 2)
    y2 = int((img.shape[1] + model.shape[1]) / 2)

    im[x1:x2, y1:y2] = model_in

    center_rotation = (im.shape[0] / 2 - 0.5, im.shape[1] / 2 - 0.5)

    matrix1 = cv2.getRotationMatrix2D(center_rotation, angle1, scale)
    im_rotated1 = cv2.warpAffine(im, matrix1, im.shape)

    matrix2 = cv2.getRotationMatrix2D(center_rotation, angle2, scale)
    im_rotated2 = cv2.warpAffine(im, matrix2, im.shape)

    im_fft_1 = np.fft.fft2(im_rotated1)
    im_fft_2 = np.fft.fft2(im_rotated2)
    img_fft = np.fft.fft2(img)

    conjugation1 = np.conj(im_fft_1) * img_fft
    conjugation1 = conjugation1 / np.abs(conjugation1)
    module_1 = np.abs(np.fft.ifft2(conjugation1))

    conjugation2 = np.conj(im_fft_2) * img_fft
    conjugation2 = conjugation2 / np.abs(conjugation2)
    module_2 = np.abs(np.fft.ifft2(conjugation2))

    if np.amax(module_1) > np.amax(module_2):
        module_m = module_1
        pattern_m = im_rotated1
    else:
        module_m = module_2
        pattern_m = im_rotated2

    y, x = np.unravel_index(np.argmax(module_m), module_m.shape)
    y = y - img.shape[1]

    print('Wektory przesunięcia x, y -> obrazy przesunięte:', x, y)
    matrix = np.float32([[1, 0, x], [0, 1, y]])
    im_rotated = cv2.warpAffine(pattern_m, matrix, (img.shape[1], img.shape[0]))

    plt.figure()
    plt.imshow(im_rotated, cmap='gray')
    plt.axis('off')
    plt.title('Model after processing')
    plt.show()

    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title('Analyzed image')
    plt.show()
