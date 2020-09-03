from PIL import Image
import numpy as np
from tqdm import tqdm


class DataManager:
    """
    color-helper for visualizations of the platelet dataset
    """

    name_class_map = {
        "Background": 0,
        "Cytoplasm": 1,
        "Organelles": 2,
    }
    color_class_map = {
        0: (0, 40, 97),
        1: (0, 40, 255),
        2: (255, 229, 0),
    }
    class_frequency_map = {
        0: 11541800 / 32000000,
        1: 17687776 / 32000000,
        2: 2770424 / 32000000,
    }

    # make dicts bidirectional
    revd = dict([reversed(i) for i in name_class_map.items()])
    name_class_map.update(revd)
    revd = dict([reversed(i) for i in color_class_map.items()])
    color_class_map.update(revd)

    def name(self, obj):
        if isinstance(obj, int):
            # class to name lookup
            return self.name_class_map[obj]
        else:
            # color to name lookup
            return self.name_class_map[self.color_class_map[tuple(obj[:3])]]

    def color(self, obj):
        if isinstance(obj, int):
            # class to color lookup
            return self.color_class_map[obj]
        else:
            # name to color lookup
            return np.array(self.color(self.name_class_map[obj]))

    def cls(self, obj):
        if isinstance(obj, str):
            # name to class lookup
            return self.name_class_map[obj]
        else:
            # color to class lookup
            return self.color_class_map[tuple(obj[:3])]

    def frequency(self, cls):
        return self.class_frequency_map[cls]


def rgb_to_class(pil_image):
    """
    transforms an RGB segmentation image to a class-map
    Parameters:
        pil_image: PIL image
    Returs:
        one-hot label map of size H x W
    """
    rgb_array = np.array(pil_image)
    # create class map
    class_labels = np.zeros(rgb_array.shape[:-1], dtype=np.uint8)

    cm = DataManager()

    # run over image
    for h in range(rgb_array.shape[0]):
        for w in range(rgb_array.shape[1]):
            pix = rgb_array[h, w]
            class_labels[h, w] = cm.cls(pix)
    return class_labels


def rgb_to_onehot(pil_image):
    """
    transforms an RGB segmentation image to one-hot encoding
    Parameters:
        pil_image: PIL image
    Returs:
        one-hot label map of size H x W x 7
    """
    rgb_array = np.array(pil_image)
    # create class map
    class_labels_onehot = np.zeros((*rgb_array.shape[:-1], 7), dtype=np.uint8)

    cm = DataManager()

    # run over image
    for h in range(rgb_array.shape[0]):
        for w in range(rgb_array.shape[1]):
            pix = rgb_array[h, w]
            c = cm.cls(pix)
            class_labels_onehot[h, w, c] = 1
    return class_labels_onehot


def class_to_rgb(class_array):
    """
    transforms a one-hot aray into an RGB segmentation image
    Parameters:
        class_array: np.array of size H x W
    Returs:
        PIL image of size H x W
    """
    # create class map
    rgb_img = np.zeros((*class_array.shape[:2], 3), dtype=np.uint8)

    cm = DataManager()
    # run over image
    for h in range(class_array.shape[0]):
        for w in range(class_array.shape[1]):
            c = class_array[h, w]
            color = cm.color(c)
            rgb_img[h, w, :] = color
    return Image.fromarray(rgb_img)


def onehot_to_rgb(onehot):
    """
    transforms a one-hot aray into an RGB segmentation image
    Parameters:
        onehot: np.array of size H x W x 7
    Returs:
        PIL image of size H x W
    """
    # create class map
    rgb_img = np.zeros((*onehot.shape[:-1], 3), dtype=np.uint8)

    cm = DataManager()
    # run over image
    for h in range(onehot.shape[0]):
        for w in range(onehot.shape[1]):
            pix = onehot[h, w]
            c = np.argmax(pix)
            color = cm.color(c)
            rgb_img[h, w, :] = color
    return Image.fromarray(rgb_img)


def transform_tiff_stack(
    all_classes, slice_no, reduced_classes_savepath, reduced_classes_rgb_savepath
):
    """
    transforms a RGB tif image stack into one hot and class images
    Parameters:
        onehot: np.array of size H x W x 7
    Returs:
        PIL image of size H x W
    """
    image_stack = Image.open(all_classes)
    class_images = []
    rgb_images = []
    for i in tqdm(range(slice_no)):
        image_stack.seek(i)
        img = np.array(image_stack)
        img[img > 1] = 2
        class_images.append(Image.fromarray(img))
        rgb_images.append(class_to_rgb(img))

    class_images[0].save(
        reduced_classes_savepath, save_all=True, append_images=class_images[1:]
    )
    rgb_images[0].save(
        reduced_classes_rgb_savepath, save_all=True, append_images=rgb_images[1:]
    )


if __name__ == "__main__":
    # preprocess data
    transform_tiff_stack(
        "./data/platelet_em_reduced/labels-class-all/24-class.tif",
        24,
        "./data/platelet_em_reduced/labels-class/24-class.tif",
        "./data/platelet_em_reduced/labels-semantic/24-semantic.tif",
    )
    transform_tiff_stack(
        "./data/platelet_em_reduced/labels-class-all/50-class.tif",
        50,
        "./data/platelet_em_reduced/labels-class/50-class.tif",
        "./data/platelet_em_reduced/labels-semantic/50-semantic.tif",
    )
