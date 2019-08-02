def obj_area_filter(bot, obj):
    return np.where(obj==255, 0, bot)

        
def draw_contours(img, contours, area):
    width = int(np.sqrt(area))
    return cv2.drawContours(img, contours, -1, 127, width) 
            


def get_obj_bbox(img):
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


# use separated mask for inpainting (this mode is failed)
if self.config.mask is 'separate':
    inputs = self._separated_mask(img, segmap)
    inpainteds = []
    for input in inputs:
        if input['area'] <= 400:
            inpainted = cv2.inpaint(input['img'], input['mask'], 3, cv2.INPAINT_NS)
        else:
            inpainted, inpainted_edge = self._inpaint(input['img'], input['mask'])
        inpainteds.append(inpainted)
    
    out = self._integrate_outputs(img, inputs, inpainteds)
    self.debugger.img(out, 'Final Output')
    out = Image.fromarray(out)
    out.save('./data/exp/cityscapes_testset/{}_sep_out_resized_{}.{}'.format(
        self.fname, self.config.resize_factor, self.fext))

def _separated_mask(self, img, semseg_map):
    # create mask image and image with mask
    print('===== Create separated inputs =====')
    inputs = self._exec_module(self.config.mask_mode, 'sep_inputs',
            self.mc.separated_mask, img, semseg_map)
    return inputs

def _integrate_outputs(self, img, inputs, outputs):
    self.debugger.img(img, 'original image')
    for input, output in zip(inputs, outputs):
        box = input['box']
        origin_wh = (box[2] - box[0], box[3] - box[1])
        output = cv2.resize(output, origin_wh)
        mask_part = cv2.resize(input['mask'], origin_wh)
        self.debugger.img(output, 'output')

        out = np.zeros(img.shape, dtype=np.uint8)
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        out[box[1]:box[3], box[0]:box[2], :] = output
        self.debugger.img(out, 'out')
        mask[box[1]:box[3], box[0]:box[2]] = mask_part
        
        mask = np.stack([mask for _ in range(3)], axis=-1)
        img = np.where(mask==255, out, img)
        self.debugger.img(img, 'img')

    return img

