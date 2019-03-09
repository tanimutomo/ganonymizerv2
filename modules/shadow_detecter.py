import cv2
import numpy as np
import torch
from skimage.segmentation import felzenszwalb, slic
from skimage.segmentation import mark_boundaries

class ShadowDetecter:
    def __init__(self, cofig, debugger):
        self.debugger = debugger
        self.thresh = 3

    
    def mask(self, img, segmap, labels):
        # collect dynamic object's id
        dynamic_object_ids = []
        for label in labels:
            if (label.category is 'vehicle' or label.category is 'human') and label.trainId is not 19:
                dynamic_object_ids.append(label.trainId)

        # create the mask image using only segmap
        obj_mask = segmap.copy()
        for do_id in dynamic_object_ids:
            obj_mask = np.where(obj_mask==do_id, 255, obj_mask).astype(np.uint8)
        obj_mask = np.where(obj_mask==255, 255, 0)

        # expnad mask area by contours
        obj_mask = self._expand_mask(obj_mask)

        # visualization
        # self.debugger.matrix(obj_mask, 'Mask Image based on only segmap')
        # self.debugger.img(obj_mask, 'Mask Image based on only segmap', gray=True)

        # image with mask
        obj_mask_3c = np.stack([obj_mask for _ in range(3)], axis=-1)
        img_with_mask = np.where(obj_mask_3c==255, obj_mask_3c, img).astype(np.uint8)
        # self.debugger.img(img_with_mask, 'Image with Mask')
        # self.debugger.matrix(img_with_mask, 'image with mask')

        return obj_mask.astype(np.uint8), img_with_mask.astype(np.uint8)


    def separated_mask(self, img, segmap, crop_rate, labels):
        mask, _ = self.mask(img, segmap, labels)
        H, W = mask.shape
        label_map, stats, labels = self._detect_object(mask)
        self.debugger.matrix(label_map, 'label map')
        self.debugger.img(label_map, 'label map')
        inputs = []
        
        for label, stat in zip(labels, stats):
            box = np.array([0, 0, 0, 0])
            obj = np.where(label_map==label, 255, 0).astype(np.uint8)
            self.debugger.matrix(obj, 'object image')
            self.debugger.img(obj, 'object image', gray=True)
            tl = np.array([stat[0], stat[1]])
            w, h = stat[2], stat[3]
            br = np.array([stat[0] + w, stat[1] + h])
            center = np.floor(((br + tl) / 2)).astype(np.int32)
            area = stat[4]

            if w*3 > 256:
                box[0], box[2] = self._crop_obj(tl[0], br[0],
                        int(w / crop_rate), 0, W)

            else:
                box[0], box[2] = self._crop_obj(center[0], center[0], 128, 0, W)
                    
            if h*3 > 256:
                box[1], box[3] = self._crop_obj(tl[1], br[1],
                        int(h / crop_rate), 0, H)

            else:
                box[1], box[3] = self._crop_obj(center[1], center[1], 128, 0, H)

            img_part = img[box[1]:box[3], box[0]:box[2], :]
            mask_part = obj[box[1]:box[3], box[0]:box[2]]

            self.debugger.matrix(img_part, 'before img part')
            # self.debugger.img(img_part, 'before img part')
            self.debugger.matrix(mask_part, 'before mask part')
            # self.debugger.img(mask_part, 'before mask part')

            img_part = self._adjust_imsize(img_part)
            mask_part = self._adjust_imsize(mask_part)

            self.debugger.matrix(img_part, 'after img part')
            # self.debugger.img(img_part, 'after img part')
            self.debugger.matrix(mask_part, 'after mask part')
            # self.debugger.img(mask_part, 'after mask part')

            assert (img_part.shape[0], img_part.shape[1]) == (mask_part.shape[0], mask_part.shape[1])
            inputs.append({'img': img_part, 'mask': mask_part, 
                'box': box, 'new_size': mask_part.shape, 'area': area})

            self.debugger.matrix(img_part, 'img part')
            self.debugger.matrix(mask_part, 'mask part')

        return inputs


    def detect(self, img, mask, img_with_mask):
        # connected components
        label_map, stats, labels = self._detect_object(mask)

        for label, stat in zip(labels, stats):
            obj_img = np.where(label_map==label, 255, 0).astype(np.uint8)
            contours = self._extract_obj_bot(obj_img)
            bot_img = np.zeros(obj_img.shape, dtype=np.uint8)
            bot_img = self._draw_contours_as_point(bot_img, contours, stat[4])
            bot_img = self._obj_area_filter(bot_img, obj_img)
 

    def _crop_obj(self, top, bot, base, min_thresh, max_thresh):
        if top - base >= min_thresh:
            ntop = top - base
            extra = 0
        else:
            ntop = min_thresh
            extra = min_thresh - (top - base)

        if bot + base + extra < max_thresh:
            nbot = bot + base + extra
            extra = 0
        else:
            nbot = max_thresh
            extra = bot + base + extra - max_thresh

        ntop = ntop - extra if ntop - extra >= min_thresh else min_thresh

        return ntop, nbot

    
    def _adjust_imsize(self, img):
        size = np.array([img.shape[0], img.shape[1]])
        nsize = np.array([0, 0])
        argmax = np.argmax(size)
        if size[argmax] > 300:
            argmin = np.argmin(size)
            rate = 300 / size[argmax]
            nsize[argmax] = 300
            nsize[argmin] = np.ceil(size[argmin] * rate).astype(np.int32)
        else:
            nsize = size.copy()

        nsize[0] = nsize[0] if nsize[0] % 4 == 0 else nsize[0] - (nsize[0] % 4)
        nsize[1] = nsize[1] if nsize[1] % 4 == 0 else nsize[1] - (nsize[1] % 4)
        out = cv2.resize(img, (nsize[1], nsize[0]))

        return out


    def _obj_area_filter(self, bot, obj):
        return np.where(obj==255, 0, bot)

        
    def _extract_obj_bot(self, img):
        H, W = img.shape
        # find contours
        contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # filtering by upside and downside of the contours
        filtered_contours = []
        for cnt in contours:
            self.debugger.matrix(cnt, 'cnt')
            cnt = np.squeeze(cnt) # cnt's shape is (num, 2(x, y))

            # upside filtering
            upside_y = cnt[:, 1] - self.thresh
            upside_y = np.where(upside_y < 0, 0, upside_y)
            upside = np.stack([cnt[:, 0], upside_y], axis=-1) # y = y - 1

            up_idx = (upside[:, 1], upside[:, 0]) # (ys, xs)
            up_pixel = img[(up_idx)]

            up_idx = np.where(up_pixel==255)
            up_filtered_cnt = cnt[up_idx]


            # downside filtering
            downside_y = up_filtered_cnt[:, 1] + self.thresh
            downside_y = np.where(downside_y >= H, H - 1, downside_y)
            downside = np.stack([up_filtered_cnt[:, 1], 
                downside_y], axis=-1) # y = y + 1

            down_idx = (downside[:, 1], downside[:, 0]) # (ys, xs)
            down_pixel = img[(down_idx)]

            down_idx = np.where(down_pixel==0)
            up_down_filtered_cnt = up_filtered_cnt[down_idx]

            # upside and downside filtering
            up_down_filtered_cnt = np.expand_dims(up_down_filtered_cnt, axis=1)
            filtered_contours.append(up_down_filtered_cnt)

        return filtered_contours
            
        
    def _draw_contours_as_point(self, img, contours, area):
        # radius = int(np.sqrt(area) / 2)
        radius = 1
        # img_with_bottom = cv2.drawContours(img, new_contours, -1, 127, width) 
        for cnt in contours:
            cnt = np.squeeze(cnt)
            for point in cnt:
                point = (point[0], point[1])
                img_with_bottom = cv2.circle(img, point, radius, 127, thickness=-1)

        return img_with_bottom


    def _draw_contours(self, img, contours, area):
        width = int(np.sqrt(area))
        return cv2.drawContours(img, contours, -1, 127, width) 
            

    def _detect_object(self, img):
        num, label_map, stats, _ = cv2.connectedComponentsWithStats(img)
        # stat is [tl_x, tl_y, w, h, area]
        label_list = [i+1 for i in range(num - 1)]
        return label_map, stats[1:], label_list


    def old_detect(self, img, mask, img_with_mask):
        # extract bottom part of the image with mask
        H, W = mask.shape[0], mask.shape[1]
        for box in bboxes:
            half_w, half_h = int((box[2] - box[0]) / 2), int((box[3] - box[1]) / 2)
            box[0] = box[0] - half_w if box[0] - half_w >= 0 else 0
            box[1] = box[1] + half_h if box[1] + half_h < H else H - 2 # half of the car height
            box[2] = box[2] + half_w if box[2] + half_w < W else W - 1
            box[3] = box[3] + half_h if box[3] + half_h < H else H - 1

        # check extracted bboxes area
        tmp = mask.astype(np.uint8).copy()
        for box in bboxes:
            tmp = cv2.rectangle(tmp, (box[0], box[1]), (box[2], box[3]), 255, 1)

        # visualization
        self.debug.img(out2, 'expanded bounding boxes')


        cars = []
        for box in bboxes:
            car = img_with_mask[box[1]:box[3], box[0]:box[2], :]
            self.debug.img(car, 'car part')
            cars.append(car)

        # the pixel of car part
        car_y, car_x = np.where(car_map==255)
        car_pos = np.array([car_x, car_y])

        # apply SLIC to car part images
        for car in cars:
            segmap, num = self._slic(car)
            segmap = segmap + 1
            car_gray = car.astype(np.int64)
            car_gray = np.sum(car_gray, axis=-1)
            self.debug.matrix(car_gray, 'car')
            segmap = np.where(car_gray==255*3, 0, segmap)
            self.debug.matrix(segmap, 'segmap')
            for idx in range(num):
                y, x = np.where(segmap==idx)
                segment_mean_color = np.sum(np.sum(car_gray[y][:, x], axis=0), axis=0)
                self.debug.matrix(segment_mean_color, 'segment mean color')


    def _slic(self, img):
        num_segments = int(img.shape[0] * img.shape[1] / 500)
        segments_slic = slic(img, n_segments=num_segments, compactness=10, sigma=1)
        print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))

        segslic_img = mark_boundaries(img, segments_slic)
        self.debug.img(segslic_img, 'SLIC Segmentation')
        self.debug.matrix(segments_slic, 'segments_slic')

        return segments_slic, num_segments


    def _expand_mask(self, mask):
        mask = mask.astype(np.uint8)
        width = int((mask.shape[0] + mask.shape[1]) / 256)
        self.debugger.param(width, 'mask expand width')
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        mask = cv2.drawContours(mask, contours, -1, 255, width) 
        
        return mask.astype(np.uint8)


    def _get_obj_bbox(self, img):
        out = img.astype(np.uint8)
        contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = []
        thresh = 10000
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            out = cv2.rectangle(out, (x, y), (x+w, y+h), 255, 2)
            if w*h >= thresh:
                bboxes.append([x, y, x+w, y+h])

        return out, bboxes


