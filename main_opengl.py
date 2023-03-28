# THIS FILE TRIES TO SIMULATE REAL-LIFE CAMERA EFFECTS ON COMPUTER-GENERATED AIRCRAFT IMAGERY
# EFFECTS IN USE:
#  [- 1st pass JPEG compression]
#   - gaussian distributed noise
#   - chromatic abberation and lens blurring
#   - bloom
#   - purple fringing
#   - weak embossing as simulation of parasitic voltage in the sensor
#   - inverse dynamic range transform
#   - thermal CCD sensor noise and ISO noise
#   - 2nd pass JPEG compression
#

from email.mime import image
from fileinput import close
import time
from ctypes import *
import multiprocessing
import asyncio
import random
import os
import io

import imageio
import numpy as np
import numpy.ctypeslib as npct
import cv2




from Unknown6656.CVGLPixelShader import PixelShader, ShaderVariable, ShaderVariableType




original = cv2.imread('img/example-in-2.png')
radius = ShaderVariable('radius', ShaderVariableType.INT)
horizontal = ShaderVariable('horizontal', ShaderVariableType.BOOL)
with PixelShader('''
float TAU = 6.28318530718;
float Directions = 16.0;
float Quality = 3.0;
float Size = 8.0;

vec2 Radius = Size / vec2(width, height);

out_color = texture2D(image, coords);

for (float phi = 0.0; phi < TAU; phi += TAU / Directions)
    for(float i = 1.0 / Quality; i <= 1.0; i += 1.0 / Quality)
        out_color += texture2D(image, coords + vec2(cos(phi), sin(phi)) * Radius * i);

out_color /= Quality * Directions;
''', '', [radius, horizontal]) as shader:
    shader[radius] = 1
    shader[horizontal] = True
    processed = shader(original)
    cv2.imwrite('mono_crash.png', processed)

exit()




from main import *
import main

main.apply_bloom = None
main.apply_chromatic_abberation = None
main.apply_embossing = None
main.apply_gauss_noise = None
main.apply_inverse_dynamic_range = None
main.apply_jpeg_compression = None
main.apply_purple_fringing = None
main.apply_sensor_noise = None



def apply_gauss_noise(image : np.ndarray, noise_intensities : list[float]) -> np.ndarray:
    width = image.shape[0]
    height = image.shape[1]

    for i in range(len(noise_intensities)):
        noise = np.random.normal(noise_intensities[i] * .5, noise_intensities[i], (width, height))
        image[:, :, i] += noise

    return image


def apply_jpeg_compression(image : np.ndarray, quality : int) -> np.ndarray:
    jpeg_stream = io.BytesIO()
    imageio.imwrite(jpeg_stream, image, format='jpg', quality=quality)
    jpeg_buffer = jpeg_stream.getbuffer()

    return imageio.v2.imread(jpeg_buffer, format='jpg')


if DYNAMIC_RANGE_USE_CPP:
    dll = WinDLL(DYNAMIC_RANGE_LIBRARY)
    inverse_dynamic_range = dll[DYNAMIC_RANGE_FUNCTION]
    inverse_dynamic_range.argtypes = [c_float, c_float, c_float]
    inverse_dynamic_range.restype = c_float
else:
    def inverse_dynamic_range(x : float, cap : float, amount : float) -> float:
        x -= .5
        x2 = x * x
        x3 = x2 * x
        c = .8 * np.exp(-10 * x2)
        d = 1.2 * (c * 8 * x3 + (1 - c) * (x - x3)) * cap

        return (d + .5) * amount + (x + .5) * (1 - amount)


def apply_inverse_dynamic_range(image : np.ndarray) -> np.ndarray:
    if DYNAMIC_RANGE_USE_FAST:
        cap = DYNAMIC_RANGE_CAP * 1.1
        return image * cap + .5 * (1 - cap)

    shape = image.shape
    image = image.reshape(-1)
    cores = multiprocessing.cpu_count() - 1
    slice_size = int(image.shape[0] * 1. / cores)

    async def process_slice(slice):
        size = slice_size if slice < cores - 1 else image.shape[0] - (cores - 1) * slice_size
        for index in range(size):
            index = index + slice * slice_size
            image[index] = inverse_dynamic_range(image[index], DYNAMIC_RANGE_CAP, DYNAMIC_RANGE_AMOUNT)

    start = time.time()
    loop = asyncio.get_event_loop()
    looper = asyncio.gather(*[process_slice(slice) for slice in range(cores)])
    loop.run_until_complete(looper)
    duration = time.time() - start

    return image.reshape(shape)


# INTER_LINEAR, INTER_CUBIC, INTER_LANCZOS4, INTER_AREA, INTER_NEAREST
def scale_rotate(image : np.ndarray, scale : tuple[float, float] | float = (1., 1.), center = (.5, .5), degrees : float = 0., flags = cv2.INTER_LINEAR) -> np.ndarray:
    if isinstance(scale, int):
        scale = (float(scale), float(scale))
    if isinstance(scale, float):
        scale = (scale, scale)
    elif len(scale) > 2:
        scale = scale[:2]

    scale = list(scale) + [1.]

    R = np.eye(3)
    T = np.eye(3)
    IT = np.eye(3)

    (ih, iw) = image.shape[:2]

    R[0:2] = cv2.getRotationMatrix2D(center=(0, 0), angle=-degrees, scale=1.)
    T[0:2, 2] = [iw * center[0], ih * center[1]]
    IT[0:2, 2] = [iw * -center[0], ih * -center[1]]
    S = np.diag(scale)

    M = (T @ R @ S @ IT)[0:2]

    return cv2.warpAffine(image, M, (iw, ih), flags = flags)


def apply_chromatic_abberation(image : np.ndarray) -> np.ndarray:
    scale = 23 / np.max(image.shape) if CHROMATIC_ABBERATION_SCALE_AUTO else CHROMATIC_ABBERATION_SCALE
    hsv = (image * 255).astype(np.uint8)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
    slice_width = 179.9 / CHROMATIC_ABBERATION_SLICES
    blurred = np.empty(image.shape)

    for slice in range(CHROMATIC_ABBERATION_SLICES):
        lower = np.array([int(slice_width * slice), 0, 0])
        upper = np.array([int(slice_width * (slice + 1)), 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        masked = cv2.bitwise_and(image, image, mask = mask)
        masked = scale_rotate(masked, 1 + slice / CHROMATIC_ABBERATION_SLICES * scale, flags=cv2.INTER_LINEAR)
        masked = cv2.blur(masked, (CHROMATIC_ABBERATION_BLUR_SIZE, CHROMATIC_ABBERATION_BLUR_SIZE), cv2.BORDER_DEFAULT)
        masked = masked.clip(0, 1)

        blurred += masked

    blurred = np.clip(blurred, 0, 1)
    image_r = image[:, :, 0]
    image_g = image[:, :, 1]

    image[:, :, 0] = scale_rotate(image_r, 1 + scale * .5, flags=cv2.INTER_LINEAR)
    image[:, :, 1] = scale_rotate(image_g, 1 + scale * .25, flags=cv2.INTER_LINEAR)

    return image * (1 - CHROMATIC_ABBERATION_BLUR_WEIGHT) + blurred * CHROMATIC_ABBERATION_BLUR_WEIGHT


def apply_purple_fringing(image : np.ndarray) -> np.ndarray:
    original = np.copy(image)
    image *= 255

    nsr = np.average(image, 2)
    nsr = cv2.threshold(nsr, PURPLE_FRINGING_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    nsr = cv2.blur(nsr / 255., (PURPLE_FRINGING_FIRST_PASS_BLUR, PURPLE_FRINGING_FIRST_PASS_BLUR))
    nsr = cv2.blur(nsr, (PURPLE_FRINGING_FIRST_PASS_BLUR, PURPLE_FRINGING_FIRST_PASS_BLUR))
    nsr = np.clip(nsr, 0, 1)

    rb = image[:, :, 0] - image[:, :, 2]
    bg = image[:, :, 2] - image[:, :, 1]

    cr1 = cv2.threshold(rb, PURPLE_FRINGING_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY_INV)[1]
    cr2 = cv2.threshold(bg, PURPLE_FRINGING_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    cr1 = cv2.blur(cr1 / 255, (PURPLE_FRINGING_SECOND_PASS_BLUR, PURPLE_FRINGING_SECOND_PASS_BLUR))
    cr2 = cv2.blur(cr2 / 255, (PURPLE_FRINGING_SECOND_PASS_BLUR, PURPLE_FRINGING_SECOND_PASS_BLUR))
    cr1 = np.clip(cr1, 0, 1)
    cr2 = np.clip(cr2, 0, 1)

    mask = np.clip(nsr * cr1 * cr2, 0, 1)
    mask = cv2.blur(mask, (PURPLE_FRINGING_SECOND_PASS_BLUR, PURPLE_FRINGING_SECOND_PASS_BLUR))
    mask = np.clip(mask * PURPLE_FRINGING_AMOUNT, 0, 1)
    mask = np.repeat(mask[:, :, None], 3, 2)

    mask[:, :, 0] *= PURPLE_FRINGING_COLOR[0]
    mask[:, :, 1] *= PURPLE_FRINGING_COLOR[1]
    mask[:, :, 2] *= PURPLE_FRINGING_COLOR[2]

    return np.clip(mask + original, 0, 1)


def apply_embossing(image : np.ndarray) -> np.ndarray:
    return cv2.filter2D(image, -1, np.array([
        [2 * EMBOSSING_STRENGTH, EMBOSSING_STRENGTH, 0],
        [EMBOSSING_STRENGTH,  1, -EMBOSSING_STRENGTH],
        [ 0, -EMBOSSING_STRENGTH, -2 * EMBOSSING_STRENGTH]
    ]))


def apply_bloom(image : np.ndarray) -> np.ndarray:
    hsv = (image * 255).astype(np.uint8)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV).astype(np.float32)
    _, s, v = cv2.split(hsv)
    sv = ((255 - s) * v / 255).clip(0, 255).astype(np.uint8)
    thresh = cv2.threshold(sv, BLOOM_WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)[1]

    blur = cv2.blur(thresh, (BLOOM_BLUR_SIZE, BLOOM_BLUR_SIZE)).clip(0, 255)
    blur = cv2.blur(blur, (BLOOM_BLUR_SIZE, BLOOM_BLUR_SIZE)).clip(0, 255)
    blur = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
    blur = blur * BLOOM_INTENSITY_GAIN / 255

    return np.clip(image + blur, 0, 1)


def apply_sensor_noise(image : np.ndarray) -> np.ndarray:
    random_state = np.random.RandomState(int(time.time()))

    photons = random_state.poisson(SENSOR_NOISE_PHOTON_COUNT_PPIXEL, image.shape)
    electrons_in = np.round(photons * SENSOR_NOISE_QUANTUM_EFFICIENCY)
    electrons_out = np.round(random_state.normal(0, SENSOR_NOISE_DARKNOISE_ELECTRONS, electrons_in.shape) + electrons_in)

    ediff = (electrons_in - electrons_out) / (4 * SENSOR_NOISE_SENSITIVITY) + .5

    max_adu = int(2 ** SENSOR_NOISE_BITDEPTH - 1)
    adu = (electrons_out * SENSOR_NOISE_SENSITIVITY) + SENSOR_NOISE_BASELINE_ADU
    adu[adu > max_adu] = max_adu
    adu = adu.astype(np.float64) / (SENSOR_NOISE_SENSITIVITY * max_adu)

    return np.clip(image + adu + (1 - image) * ediff * SENSOR_NOISE_EDIFF_INFLUENCE, 0, 1)


def postprocess(image : np.ndarray) -> np.ndarray:
    if CROP:
        image = image[CROP_BY[0]:(None if CROP_BY[2] is None else -CROP_BY[2]), CROP_BY[3]:(None if CROP_BY[1] is None else -CROP_BY[1]), :3]

    if JPEG_COMPRESSION_LEVEL_FIRST_PASS is not None:
        image = apply_jpeg_compression(image, JPEG_COMPRESSION_LEVEL_FIRST_PASS)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    image = np.transpose(image, (1, 0, 2))

    if GAUSS_NOISE:             image = apply_gauss_noise(image, GAUSS_NOISE_INTENSITY)
    if CHROMATIC_ABBERATION:    image = apply_chromatic_abberation(image)
    if BLOOM:                   image = apply_bloom(image)
    if PURPLE_FRINGING:         image = apply_purple_fringing(image)
    if EMBOSSING:               image = apply_embossing(image)
    if DYNAMIC_RANGE:           image = apply_inverse_dynamic_range(image)
    if SENSOR_NOISE:            image = apply_sensor_noise(image)

    image = np.transpose(image, (1, 0, 2))
    image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    if JPEG_COMPRESSION_LEVEL_SECOND_PASS is not None:
        image = apply_jpeg_compression(image, JPEG_COMPRESSION_LEVEL_SECOND_PASS)

    return image


def postprocess_batch(input_dir : str, output_dir : str) -> None:
    input_dir = os.path.normpath(input_dir)
    output_dir = os.path.normpath(output_dir)
    files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    findex = 0

    for file in files:
        findex += 1
        in_file = os.path.join(input_dir, file)
        image = cv2.imread(in_file)

        start = time.time()
        image = postprocess(image)
        duration = time.time() - start

        out_file = os.path.join(output_dir, file.replace('.png', '-processed.png') if input_dir == output_dir else file)
        cv2.imwrite(out_file, image)

        print(f'[{findex:5}/{len(files):5}] {in_file:60} {duration} s')




# EXAMPLE: 

original = cv2.imread('img/example-in-2.png')
original = postprocess(original)
cv2.imwrite('img/example-out-2.png', original)
