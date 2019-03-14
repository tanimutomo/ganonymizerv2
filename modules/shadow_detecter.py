import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from skimage import color
from skimage.segmentation import felzenszwalb, slic, quickshift, mark_boundaries
from skimage.future import graph
from sklearn.cluster import MeanShift


class ShadowDetecter:
    def __init__(self, config, debugger):
        self.config = config
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
        # obj_mask = self._expand_mask(obj_mask)

        return obj_mask.astype(np.uint8)


    def separated_mask(self, img, segmap, crop_rate, labels):
        mask, _ = self.mask(img, segmap, labels)
        H, W = mask.shape
        label_map, stats, labels = self._detect_object(mask)
        inputs = []
        
        for label, stat in zip(labels, stats):
            box = np.array([0, 0, 0, 0])
            obj = np.where(label_map==label, 255, 0).astype(np.uint8)
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

            img_part = self._adjust_imsize(img_part)
            mask_part = self._adjust_imsize(mask_part)

            assert (img_part.shape[0], img_part.shape[1]) == (mask_part.shape[0], mask_part.shape[1])
            inputs.append({'img': img_part, 'mask': mask_part, 
                'box': box, 'new_size': mask_part.shape, 'area': area})

        return inputs


    def detect(self, img, mask):
        # connected components
        label_map, stats, labels = self._detect_object(mask)
        shadow_masks = []

        for label, stat in zip(labels, stats):
            # get each object in turn
            obj_mask = np.where(label_map==label, 255, 0).astype(np.uint8)

            # object area filtering
            if stat[4] < img.shape[0] * img.shape[1] * self.config['obj_sml_thresh']:
                continue

            # calcurate the median of object size
            hor_m, ver_m = self._ver_hor_median(obj_mask)

            # get the bbox of object's bottom area
            bbox = self._get_bottom_box(obj_mask, ver_m, hor_m, img.shape[1], img.shape[0])

            # object location filtering
            if bbox[3] < img.shape[0] * self.config['obj_high_thresh']:
                continue

            # Visualize Objec Mask
            self.debugger.matrix(obj_mask, 'object mask')
            self.debugger.img(obj_mask, 'object mask')

            # image with mask
            obj_mask_3c = np.stack([obj_mask for _ in range(3)], axis=-1)
            img_with_mask = np.where(obj_mask_3c==255, obj_mask_3c, img).astype(np.uint8)

            # visualization
            self.debugger.matrix(bbox, 'bbox')
            obj_with_bot_bbox = obj_mask.copy()
            obj_with_bot_bbox = cv2.rectangle(obj_with_bot_bbox, (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]), 255, 2)
            self.debugger.img(obj_with_bot_bbox,
                    'Image of Object with Bottom Area based on object size medians')

            # extract bottom area image
            bot_img = self._box_img(img_with_mask, bbox)
            self.debugger.matrix(bot_img, 'bottom area image')
            self.debugger.img(bot_img, 'bottom area image')
            
            # QuickShift Segmentation
            segmap = self._superpixel(bot_img, self.config['superpixel'])

            # extract segment which is bottom of object
            obj_bot_mask = self._box_img(obj_mask, bbox)
            self.debugger.img(obj_bot_mask, 'Object Mask Image')
            point = self._extract_obj_bot(obj_bot_mask)
            bot_point_img = np.zeros(obj_bot_mask.shape, dtype=np.uint8)
            bot_point_img = self._draw_contours_as_point(bot_point_img, point, stat[4])
            self.debugger.img(bot_point_img, 'Point of Object Bottom Area')

            # extract bottom area segment
            bot_segmap, bot_segments_mcolor = self._extract_bot_segment(bot_img, segmap, bot_point_img)

            # visualize color distribution
            label, gray = self._color_dist(np.array(list(bot_segments_mcolor.keys())),
                    np.array(list(bot_segments_mcolor.values())))

            # clustering gray
            cluster = self._meanshift(gray.reshape(-1, 1))
            self.debugger.param(cluster, 'Belonging Cluster')
            counter = Counter(list(cluster))
            self.debugger.param(counter, 'Segment Count for each Cluster')
            clst_mean = self._cluster_mean(cluster, gray)
            self.debugger.param(clst_mean, 'Cluster Color Mean')

            # get the lowest mean cluster as shadow cluster
            shadow_clst, most_color = self._get_shadow_cluster(cluster, clst_mean, counter)

            # this object doesn't have shadow
            if shadow_clst is None:
                continue

            # visualize shadow cluster segment
            sc_segmap = self._get_sc_segmap(bot_img, bot_segmap, cluster, label, shadow_clst)
            self.debugger.matrix(sc_segmap, 'Check sc_segmap')

            # object bottom area fitering by segment's centroid
            ss_segmap, ss_labels = self._obj_bot_filter(bot_img, sc_segmap)

            # this object doesn't have shadow
            if len(ss_labels) == 0:
                continue

            # find other shadow segment using mean color and RAG
            ss_segmap, ss_labels = self._add_all_ss(bot_img, segmap, ss_labels, most_color)
            self.debugger.matrix(ss_segmap, 'shadow segment visualization')

            # create a part of shadow mask
            shadow_mask = self._mask_from_segmap(ss_segmap)
            self.debugger.img(shadow_mask, 'shadow mask')

            # add all shadow mask dict
            shadow_masks.append({'bbox': bbox, 'mask': shadow_mask})

        return self._expand_mask(
                self._combine_masks(
                    mask.shape, shadow_masks
                    )
                )


    def _combine_masks(self, shape, mask_data):
        entire_mask = np.zeros(shape, dtype=np.uint8)
        for data in mask_data:
            bbox = data['bbox']
            mask = data['mask']
            entire_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] += mask
        self.debugger.img(entire_mask, 'Entire Mask Image')
        return entire_mask


    def _mask_from_segmap(self, segmap):
        mask = segmap + 1
        mask = np.where(mask == np.max(mask), 0, 255)
        return mask.astype(np.uint8)


    def _add_all_ss(self, img, segmap, ss_labels, most_color):
        vis = mark_boundaries(img, segmap)
        self.debugger.matrix(ss_labels, 'Shadow Segment Labels')
        self._visualize_seg_labels(img, segmap)

        # create rag
        G = graph.rag_mean_color(img, segmap, mode='similarity')
        graph.show_rag(segmap, G, img)

        # create labels list and segment median colors list (convert grayscale)
        labels = []
        mcolors = []
        for node in G.nodes:
            label = G.nodes[node]['labels'][0]
            labels.append(label)
            mcolors.append(np.mean(self._segment_median_rgb(img, segmap, label)))
        labels = np.array(labels)
        mcolors = np.array(mcolors)
        self.debugger.matrix(labels, 'labels')
        self.debugger.matrix(mcolors, 'mean colors')

        # create start and end edges list
        edges = np.array(G.edges)
        s_edges = np.concatenate([edges[:, 0], edges[:, 1]])
        e_edges = np.concatenate([edges[:, 1], edges[:, 0]])
        self.debugger.matrix(s_edges, 'start edges')
        self.debugger.matrix(e_edges, 'end edges')
        
        # calcurate the mean color of shadow segment
        scolors = []
        # unused labels list for searching the segments except for ss
        unused_labels = labels.copy()
        for label in ss_labels:
            scolors.append(mcolors[np.where(labels == label)[0]])
            unused_labels = np.delete(unused_labels, np.where(unused_labels == label)[0])
        scolor = np.mean(scolors)
        self.debugger.matrix(scolor, 'mean shadow color')
        self.debugger.matrix(unused_labels, 'unused labels')

        # calcurate the allowing color range
        alw_range = min(int((most_color - scolor) / 2), self.config['alw_range_max'])
        self.debugger.param(alw_range, 'Allowing Color Range')

        # find all shadow segments
        for num in range(self.config['find_iteration']):
            add_ss_labels = []
            for slabel in ss_labels:
                e_idx = np.where(s_edges==slabel)[0]
                as_label = e_edges[e_idx]
                for label in as_label:
                    if label in unused_labels:
                        color = mcolors[np.where(labels == label)[0]]
                        if color > scolor - alw_range and color < scolor + alw_range:
                            unused_labels = np.delete(unused_labels, np.where(unused_labels == label)[0])
                            add_ss_labels.append(label)

            ss_labels.extend(add_ss_labels)
            if num != 0:
                if np.all(unused_labels == pre_unused_labels):
                    break
            pre_unused_labels = unused_labels.copy()

        self.debugger.matrix(ss_labels, 'Updated SS labels')
        ss_segmap = self._filtered_segmap(img, segmap, ss_labels)

        return ss_segmap, ss_labels


    def _obj_bot_filter(self, img, segmap):
        # create rag
        G = graph.rag_mean_color(img, segmap, mode='similarity')

        # visualize for calcurating centroid
        lc = graph.show_rag(segmap, G, img)
        cbar = plt.colorbar(lc)

        outside = np.max(segmap)
        ss_labels = []
        for node in G.nodes:
            label = G.nodes[node]['labels'][0]
            if label != outside:
                centroid = np.array(G.nodes[node]['centroid'])

                # calcurate boundingbox of segmap
                H, W = img.shape[0], img.shape[1] 
                ys, xs = np.where(segmap == label)
                box = [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]
                width, height = box[2] - box[0], box[3] - box[1]


                if centroid[0] > H * self.config['shadow_high_thresh']:
                    # create check pixels for checking whether a segment is under the object
                    hs = [int(centroid[0] * (2 / 3)), int(centroid[0] * (1 / 3)), 0]
                    ws = [
                            centroid[1],
                            centroid[1] - int(width / 2) if centroid[1] - int(width / 2) > 0 else 0,
                            centroid[1] + int(width / 2) if centroid[1] + int(width / 2) < W else W - 1
                            ]
                    checkpxs = []
                    for h in hs:
                        for w in ws:
                            checkpxs.append([h, w])
                    checkpxs = np.array(checkpxs)

                    # # visualize check pixel place
                    # vis = img.copy()
                    # for cpx in checkpxs:
                    #     vis[cpx[0], cpx[1], :] = (0, 255, 255)
                    # self.debugger.img(vis, 'check pixel visualization')

                    # calcurate the shadow segment score (ss_score)
                    ss_score = 0
                    for cpx in checkpxs:
                        rgb = np.sum(img[cpx[0], cpx[1], :])
                        if rgb == 255 * 3:
                            ss_score += 1

                    self.debugger.matrix(ss_score, 'Shadow Segment Score')
                    if ss_score > self.config['ss_score_thresh']:
                        ss_labels.append(label)

        self.debugger.matrix(ss_labels, 'Shadow Segment Labels')
        ss_segmap = self._filtered_segmap(img, segmap, ss_labels)

        return ss_segmap, ss_labels


    def _get_sc_segmap(self, img, segmap, cluster, labels, sc):
        # get shadow cluster segment's label
        idx = np.where(np.array(cluster) == sc)[0]
        sc_labels = labels[idx]
        sc_segmap = self._filtered_segmap(img, segmap, sc_labels)

        return sc_segmap


    def _filtered_segmap(self, img, segmap, labels):
        outside = np.max(segmap)
        tmp = outside + 1
        new_segmap = segmap.copy()
        for label in labels:
            new_segmap = np.where(new_segmap == label, tmp, new_segmap)
        new_segmap = np.where(new_segmap == tmp, segmap, outside)

        vis = mark_boundaries(img, new_segmap)
        self.debugger.img(vis, 'Filtered Segment Map')

        return new_segmap


    def _get_shadow_cluster(self, cluster, means, counter):
        if len(cluster) < 10: return None, None
        low_mean_clst = np.argmin(means)
        self.debugger.param(low_mean_clst, 'low_mean_clst')
        if counter[low_mean_clst] < len(cluster) * 0.05: return None, None
        count_except_sc = counter.copy()
        del count_except_sc[low_mean_clst]
        self.debugger.param(count_except_sc, 'count_except_sc:')
        most_clst = max(count_except_sc, key=count_except_sc.get)
        most_clst_mean = means[most_clst]
        self.debugger.param(most_clst, 'most_clst:')
        self.debugger.param(most_clst_mean, 'most_clst_mean:')
        if counter[most_clst] < len(cluster) * 0.2: return None, None
        if means[low_mean_clst] * self.config['sc_color_thresh'] > most_clst_mean: return None, None
        self.debugger.param(low_mean_clst, 'shadow_clst:')

        return low_mean_clst, most_clst_mean


    def _cluster_mean(self, cluster, value):
        clst_mean = []
        for clst in range(np.max(cluster)+1):
            idx = np.where(cluster==clst)[0]
            clst_mean.append(np.mean(value[idx]))

        return clst_mean


    def _visualize_seg_labels(self, img, segmap):
        vis = mark_boundaries(img, segmap)
        labels = np.unique(segmap)
        for label in labels:
            ys, xs = np.where(segmap==label)
            my, mx = np.median(ys).astype(np.int32), np.median(xs).astype(np.int32)
            font = cv2.FONT_HERSHEY_SIMPLEX
            vis = cv2.putText(vis, str(label), (mx - 10, my), font, 0.3, (0, 255, 255), 1, cv2.LINE_AA)
        self.debugger.img(vis, 'Bottom Image with Segmap Label')


    def _color_dist(self, label, color):
        gray = np.mean(color, axis=1)
        self.debugger.matrix(gray, 'Mean of Segment Median Color')
        order = np.argsort(gray)
        gray, label = gray[order], label[order]
        plt.figure(figsize=(10, 10), dpi=200)
        plt.bar(np.arange(gray.shape[0]), gray,
                tick_label=label, align='center')

        return label, gray


    def _extract_bot_segment(self, img, segmap, point_img):
        # the label of object bottom area
        ys, xs = np.where(point_img==255)
        bots = np.unique(segmap[ys, :][:, xs])

        tmp = np.max(segmap) + 1
        bot_segmap = segmap.copy()
        
        # extract bottom area segment from all segmap image
        # and create the dict of {label: median color}
        label_mcolor = {}
        for label in bots:
            rgb_median = self._segment_median_rgb(img, segmap, label)
            if np.sum(rgb_median) != 255.0 * 3.0:
                bot_segmap = np.where(bot_segmap==label, tmp, bot_segmap)
                label_mcolor[label] = rgb_median
        bot_segmap = np.where(bot_segmap==tmp, segmap, tmp)
        self.debugger.matrix(bot_segmap, 'Bottom segmap')

        # visualize segment label in the image
        self._visualize_seg_labels(img, bot_segmap)

        return bot_segmap, label_mcolor


    def _segment_median_rgb(self, img, segmap, label):
        ys, xs = np.where(segmap==label)
        if len(list(img.shape)) == 3:
            segment_color = img[ys, :, :][:, xs, :]
        else:
            segment_color = img[ys, :][:, xs]
        color_median = np.median(segment_color, axis=(0, 1))

        return color_median


    def _box_img(self, img, box):
        if len(list(img.shape)) == 3:
            return img[box[1]:box[3], box[0]:box[2], :]
        else:
            return img[box[1]:box[3], box[0]:box[2]]


    def _get_bottom_box(self, img, ver_m, hor_m, W, H):
        box = self._get_bbox(img)
        maxm = max(hor_m, ver_m)
        box[0] = box[0] - maxm if box[0] - maxm >= 0 else 0
        box[1] = box[1] + int((box[3] - box[1]) / 2)
        box[2] = box[2] + maxm if box[2] + maxm < W else W - 1
        box[3] = box[3] + maxm if box[3] + maxm < H else H - 1

        return box


    def _ver_hor_median(self, img):
        medians = []
        for img in [img, np.transpose(img, (1, 0))]:
            y, x = np.where(img==255)
            yset = list(set(y))
            yset.sort()

            lengths = []
            for ys in yset:
                yidx = np.where(y==ys)[0]
                lengths.append(len(yidx))
            length_median = np.median(np.array(lengths))
            medians.append(np.floor(length_median))

        return medians


    def _get_bbox(self, img):
        y, x = np.where(img==255)
        return np.array([np.min(x), np.min(y), np.max(x), np.max(y)])


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
            cnt = np.squeeze(cnt) # cnt's shape is (num, 2(x, y))

            # upside filtering
            upside_y = cnt[:, 1] - self.thresh
            upside_y = np.where(upside_y < 0, 0, upside_y)
            upside = np.stack([cnt[:, 0], upside_y], axis=-1) # y = y - 1

            up_idx = (upside[:, 1], upside[:, 0]) # (ys, xs)
            up_pixel = img[(up_idx)]

            up_idx = np.where(up_pixel==255)
            filtered_cnt = cnt[up_idx]


            # # downside filtering
            # downside_y = up_filtered_cnt[:, 1] + self.thresh
            # downside_y = np.where(downside_y >= H, H - 1, downside_y)
            # downside = np.stack([up_filtered_cnt[:, 1], 
            #     downside_y], axis=-1) # y = y + 1

            # down_idx = (downside[:, 1], downside[:, 0]) # (ys, xs)
            # down_pixel = img[(down_idx)]

            # down_idx = np.where(down_pixel==0)
            # filtered_cnt = filtered_cnt[down_idx]

            # upside and downside filtering
            filtered_cnt = np.expand_dims(filtered_cnt, axis=1)
            filtered_contours.append(filtered_cnt)

        return filtered_contours
            
        
    def _draw_contours_as_point(self, img, contours, area):
        # radius = int(np.sqrt(area) / 2)
        out = img.copy()
        radius = 1
        for cnt in contours:
            cnt = np.squeeze(cnt)
            for point in cnt:
                point = (point[0], point[1])
                out[point[1], point[0]] = 255
                # img_with_bottom = cv2.circle(img, point, radius, 127, thickness=-1)

        return out


    def _draw_contours(self, img, contours, area):
        width = int(np.sqrt(area))
        return cv2.drawContours(img, contours, -1, 127, width) 
            

    def _detect_object(self, img):
        num, label_map, stats, _ = cv2.connectedComponentsWithStats(img)
        # stat is [tl_x, tl_y, w, h, area]
        label_list = [i+1 for i in range(num - 1)]
        return label_map, stats[1:], label_list


    def _superpixel(self, img, method, ngc=False):
        if method == 'slic':
            num_segments = int(img.shape[0] * img.shape[1] / 100)
            segmap = slic(img, n_segments=num_segments, compactness=10, sigma=1)
        elif method == 'felzenzwalb':
            segmap = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
        elif method == 'quickshift':
            segmap = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)

        if ngc:
            g = graph.rag_mean_color(img, segmap, mode='similarity')
            segmap = graph.cut_normalized(segmap, g, thresh=0.3)
            segmented_img = mark_boundaries(img, segmap)
            self.debugger.matrix(segmap, 'Segmentation by {} followed by NGC'.format(method))
            self.debugger.img(segmented_img, 'Segmentation by {} followed by NGC'.format(method))
        else:
            segmented_img = mark_boundaries(img, segmap)
            self.debugger.matrix(segmap, 'Segmentation by {}'.format(method))
            self.debugger.img(segmented_img, 'Segmentation by {}'.format(method))


        return segmap


    def _meanshift(self, data):
        ms = MeanShift(n_jobs=-1)
        ms.fit(data)
        out = ms.labels_
        return out


    def _expand_mask(self, mask):
        mask = mask.astype(np.uint8)
        width = int((mask.shape[0] + mask.shape[1]) / 500)
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


