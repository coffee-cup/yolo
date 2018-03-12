import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import six
import skimage
from bs4 import BeautifulSoup
from skimage import io


class PascalVOC(object):
    CATEGORY_NAMES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    PRESENCE_TRUE = 1
    PRESENCE_FALSE = -1
    PRESENCE_DIFFICULT = 0

    def __init__(self, base_dir, year='2012'):
        root_dir = os.path.join(base_dir, 'VOCdevkit', 'VOC{}'.format(year))
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'JPEGImages')
        self.ann_dir = os.path.join(root_dir, 'Annotations')
        self.set_dir = os.path.join(root_dir, 'ImageSets', 'Main')

        train_path = os.path.join(self.set_dir, 'train.txt')
        train_image_names = pd.read_csv(
            train_path, delim_whitespace=True, header=None, names=['filename'])
        self.train_image_names = set(train_image_names['filename'].values)

        val_path = os.path.join(self.set_dir, 'val.txt')
        val_image_names = pd.read_csv(
            val_path, delim_whitespace=True, header=None, names=['filename'])
        self.val_image_names = set(val_image_names['filename'].values)

        trainval_path = os.path.join(self.set_dir, 'trainval.txt')
        trainval_image_names = pd.read_csv(
            trainval_path,
            delim_whitespace=True,
            header=None,
            names=['filename'])
        self.trainval_image_names = set(
            trainval_image_names['filename'].values)

        self.all_image_names = set([
            os.path.splitext(filename)[0]
            for filename in os.listdir(self.img_dir)
        ])

        self.__all_annotations = None
        self.__all_image_metadata = None

    @staticmethod
    def _read_image_list(path):

        names = [line.strip() for line in open(path).readlines()]
        return [name for name in names if name != '']

    def imgs_from_category(self, category, dataset):
        """
        Summary

        Args:
            category (string): Category name as a string (from CLASS_NAMES)
            dataset (string): "train", "val", "train_val", or "test" (if available)

        Returns:
            pandas dataframe: pandas DataFrame containing a row for each image,
            the first column 'filename' gives the name of the image while the second
            column 'presence' gives one of PRESENCE_TRUE (object present in image),
            PRESENCE_FALSE (object not present in image) or PRESENCE_DIFFICULT
            (object visible but difficult recognise without substantial use of context)
        """
        if category not in self.CATEGORY_NAMES:
            raise ValueError('Unknown category {}'.format(category))
        filename = os.path.join(self.set_dir,
                                category + "_" + dataset + ".txt")
        df = pd.read_csv(
            filename,
            delim_whitespace=True,
            header=None,
            names=['filename', 'presence'])
        return df

    def _get_image_names(self, image_names, category, dataset):
        datasets_by_name = {
            'train': self.train_image_names,
            'val': self.val_image_names,
            'trainval': self.trainval_image_names
        }
        if image_names is not None:
            if isinstance(image_names, six.string_types):
                image_names = [image_names]
        else:
            if dataset is None and category is None:
                image_names = self.trainval_image_names
            else:
                if category is None:
                    image_names = datasets_by_name[dataset]
                else:
                    if dataset is None:
                        dataset = 'trainval'
                    imgs = self.imgs_from_category(category, dataset)
                    imgs = imgs[imgs['presence'] == self.PRESENCE_TRUE]
                    image_names = list(imgs['filename'].values)
        return image_names

    def __annotation_path(self, img_name):
        """
        Given an image name, get the annotation file for that image

        Args:
            img_name (string): string of the image name, relative to
                the image directory.

        Returns:
            string: file path to the annotation file
        """
        img_name = os.path.splitext(img_name)[0]
        return os.path.join(self.ann_dir, img_name) + '.xml'

    def __get_annotation_xml(self, img_filename):
        """
        Load annotation file for a given image.

        Args:
            img_name (string): string of the image name, relative to
                the image directory.

        Returns:
            BeautifulSoup structure: the annotation labels loaded as a
                BeautifulSoup data structure
        """
        xml = ""
        with open(self.__annotation_path(img_filename)) as f:
            xml = f.readlines()
        xml = ''.join([line.strip('\t') for line in xml])
        return BeautifulSoup(xml)

    def __read_all_annotations(self):
        if self.__all_annotations is None or self.__all_image_metadata is None:
            # Not loaded; see if the dataframe files exists
            annotations_df_path = os.path.join(self.ann_dir,
                                               'all_annotations.h5')
            img_meta_df_path = os.path.join(self.ann_dir, 'all_metadata.h5')
            if not os.path.exists(annotations_df_path) or not os.path.exists(
                    img_meta_df_path):
                # No; load the XML annotations and cache
                anno_data = []
                img_meta = []

                for img_i, img_name in enumerate(self.all_image_names):
                    anno = self.__get_annotation_xml(img_name)

                    train = img_name in self.train_image_names
                    val = img_name in self.val_image_names
                    trainval = img_name in self.trainval_image_names

                    fname = str(anno.findChild('filename').contents[0])
                    objs = anno.findAll('object')
                    for obj in objs:
                        obj_names = obj.findChildren('name')
                        cat = str(obj_names[0].contents[0])
                        bbox = obj.findChildren('bndbox')[0]
                        xmin = float(bbox.findChildren('xmin')[0].contents[0])
                        ymin = float(bbox.findChildren('ymin')[0].contents[0])
                        xmax = float(bbox.findChildren('xmax')[0].contents[0])
                        ymax = float(bbox.findChildren('ymax')[0].contents[0])
                        occ_xml = obj.findChildren('occluded')
                        trunc_xml = obj.findChildren('truncated')
                        diff_xml = obj.findChildren('difficult')
                        occluded = truncated = difficult = False
                        if len(occ_xml) > 0:
                            occluded = bool(int(occ_xml[0].contents[0]))
                        if len(trunc_xml) > 0:
                            truncated = bool(int(trunc_xml[0].contents[0]))
                        if len(diff_xml) > 0:
                            difficult = bool(int(diff_xml[0].contents[0]))
                        anno_data.append([
                            fname, cat, xmin, ymin, xmax, ymax, occluded,
                            truncated, difficult, train, val, trainval
                        ])

                    size = anno.findChild('size')
                    width = int(size.findChildren('width')[0].contents[0])
                    height = int(size.findChildren('height')[0].contents[0])
                    depth = int(size.findChildren('depth')[0].contents[0])
                    img_meta.append(
                        [fname, width, height, depth, train, val, trainval])

                    sys.stdout.write('\r{}'.format(img_i))

                anno_df = pd.DataFrame(
                    anno_data,
                    columns=[
                        'image_filename', 'category', 'xmin', 'ymin', 'xmax',
                        'ymax', 'occluded', 'truncated', 'difficult', 'train',
                        'val', 'trainval'
                    ])

                img_meta_df = pd.DataFrame(
                    img_meta,
                    columns=[
                        'image_filename', 'width', 'height', 'depth', 'train',
                        'val', 'trainval'
                    ])

                anno_df.to_hdf(annotations_df_path, 'annotations')
                img_meta_df.to_hdf(img_meta_df_path, 'img_metadata')

                self.__all_annotations = anno_df
                self.__all_image_metadata = img_meta_df
            else:
                self.__all_annotations = pd.read_hdf(annotations_df_path,
                                                     'annotations')
                self.__all_image_metadata = pd.read_hdf(
                    img_meta_df_path, 'img_metadata')

    @property
    def all_annotations(self):
        self.__read_all_annotations()
        return self.__all_annotations

    @property
    def all_image_metadata(self):
        self.__read_all_annotations()
        return self.__all_image_metadata

    def get_annotations(self, image_names=None, category=None, dataset=None):
        df = self.all_annotations
        df = self.__dataset_filter(df, dataset)
        df = self.__category_filter(df, category)
        df = self.__image_names_filter(df, image_names)
        return df

    def get_image_metadata(self, image_names=None, category=None,
                           dataset=None):
        df = self.all_image_metadata
        df = self.__image_names_filter(df, image_names)
        if image_names is not None:
            df = df[df.apply(
                lambda x: x['image_filename'] in image_names, axis=1)]
        return df

    def load_img(self, img_filename):
        """
        Load image from the filename. Default is to load in color if
        possible.

        Args:
            img_name (string): string of the image name, relative to
                the image directory.

        Returns:
            np array of float32: an image as a numpy array of float32
        """
        if os.path.splitext(img_filename)[1] == '':
            img_filename = img_filename + '.jpg'
        img_filename = os.path.join(self.img_dir, img_filename)
        img = skimage.img_as_float(io.imread(img_filename)).astype(np.float32)
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        return img

    def __image_names_filter(self, df, image_names=None):
        if image_names is None:
            return df
        else:
            return df[df.apply(
                lambda x: x['image_filename'] in image_names, axis=1)]

    def __category_filter(self, df, category=None):
        if category is None:
            return df
        elif category in self.CATEGORY_NAMES:
            return df[df['category'] == category]
        else:
            raise ValueError('Invalid category \'{}\''.format(category))

    def __dataset_filter(self, df, dataset=None):
        if dataset is None:
            return df
        elif dataset == 'train':
            return df[df['train'] == True]
        elif dataset == 'val':
            return df[df['val'] == True]
        elif dataset == 'trainval':
            return df[df['trainval'] == True]
        else:
            raise ValueError(
                'Invalid dataset \'{}\'; valid values are train, val or trainval'.
                format(dataset))
