import os 
def get_config(path):
    config = {
            # execution setting
            'checkpoint': os.path.join(path, 'ckpt'),
            'output': os.path.join(path, 'output'),
            'cuda': 0,

            # *_mode (choose in ['pass', 'save', 'exec', 'debug', 'none'])
            'main_mode': 'exec',
            'semseg_mode': 'exec',
            'mask_mode': 'exec',
            'split_mode': 'exec',
            'shadow_mode': 'exec',
            'divide_mode': 'exec',
            'inpaint_mode': 'exec',
            # evaluate pmd
            'random_mode': 'none', 

            # evaluate pmd
            'eval_pmd_path': os.path.join(path, 'pmd'),
            'rmask_min': 200,
            'rmask_max': 400,
            'rmask_shape': 'rectangle',

            # resize
            'resize_factor': 1,
        
            # segmentation
            'semseg': 'DeepLabV3',
            'resnet': 18,

            # mask
            'mask': 'entire',
            'expand_width': 6,
            # separate
            'crop_rate': 0.5,
            
            # separate mask to each object
            'fill_hole': 'later',
            'obj_sml_thresh': 1e-3, # this param is also used in shadow detection
            'obj_sep_thresh': 1/3,

            # shadow detection
            'obj_high_thresh': 0.2,
            'superpixel': 'quickshift',
            'shadow_high_thresh': 0.01,
            'alw_range_max': 15,
            'find_iteration': 3,
            'ss_score_thresh': 4,
            'sc_color_thresh': 1.5,

            # pseudo mask division
            # pmd mode is choosen from ['all', 'lattice', 'center', 'dprob']
            'pmd': 'all',
            'obj_wh_thresh': 120,
            'obj_density_thresh': 0.4,
            'line_width_div': 8,
            'distance': 20,

            # inpaint
            'inpaint': 'EdgeConnect',
            'sigma': 2 # for canny edge detection
            }
    return config
