import os, cv2, copy, argparse, json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--refexp_path', required=True, type=str)
parser.add_argument('--scene_path',  required=True, type=str)
parser.add_argument('--img_dir_path', required=True, type=str)
parser.add_argument('--num_refexps',  type=int, default=10)
parser.add_argument('--get_mask', action='store_true', default=False)
parser.add_argument('--get_box', action='store_true', default=False)
args = parser.parse_args()

class Dataset:
    def __init__(self, scene_file, refexp_path):
        self.scene_file = scene_file
        self.refexp_path = refexp_path
        self.scenes = None
        self.rexps = None


    def load_scene_refexp(self):
        print('loading scene.json...')
        scenes = json.load(open(self.scene_file))
        self.scenes = scenes['scenes']
        print('loading refexp.json...')
        self.rexps = json.load(open(self.refexp_path))['refexps'][:]
        print('loading json done')

        self.imgid_scenes={}
        for sce in self.scenes:
            img_id = sce['image_index']
            self.imgid_scenes[img_id] = sce


    def get_refexps(self):
        return self.rexps


    def get_scene_of_refexp(self, rexp):
        image_index = rexp['image_index']
        sce = self.imgid_scenes[image_index]
        return sce


    def get_refexp_output_objectlist(self, rexp):
        prog = rexp['program']
        image_filename = rexp['image_filename']
        last = prog[-1]
        obj_list = last['_output']
        return obj_list


    def get_mask_from_objlist(self, anno_obj, rexp, height=-1, width=-1):
        sce = self.get_scene_of_refexp(rexp)
        assert type(anno_obj) is list
        def str_to_biimg(imgstr):
            img=[]
            cur = 0
            for num in imgstr.strip().split(','):
                num = int(num)
                img += [cur] * num
                cur = 1 - cur
            return np.array(img)

        mask_pixel_num = 480*320
        gt_mask  = np.array([0]*mask_pixel_num)
        obj_mask = sce['obj_mask']
        for one_obj in anno_obj:
            gt_mask |= str_to_biimg(obj_mask[str(int(one_obj) + 1)])

        if height == -1 and width == -1:
            return gt_mask.reshape((320,480))
        else:
            return cv2.resize(gt_mask.reshape((320,480)).astype(np.uint8), (width, height))


def draw_img_with_box(img, box_list):
    import copy
    for box in box_list:
        cv2.rectangle(img,(box[0], box[1]),(box[0] + box[2], box[1] + box[3]),(0,139,69),2)
    return img 

def get_figure(args):
    img_dir = args.img_dir_path

    dset = Dataset(args.scene_path, args.refexp_path) 
    dset.load_scene_refexp()

    cum_I = 0; cum_U = 0 
    acc = 0; all_num = 0 
    for _i, rexp in enumerate(dset.get_refexps()):
        if _i > args.num_refexps:
            break

        if _i % 5000 == 0:
            print 'process', _i

        sce = dset.get_scene_of_refexp(rexp)
        rexp_text = str(rexp['refexp'])

        output_obj_idxs = dset.get_refexp_output_objectlist(rexp)
        mask = dset.get_mask_from_objlist(output_obj_idxs, rexp)

        # im_seg 
        if args.get_mask:
            image_fn = rexp['image'] + '.png'
            img_path = os.path.join(img_dir, image_fn)
            im = cv2.imread(img_path)

            mask = mask.astype('uint8')
            im_seg = im / 2 
            c2 = im_seg[:, :, 2]
            for ci in range(c2.shape[0]):
                for cj in range(c2.shape[1]):
                    if mask[ci][cj] :
                        im_seg[ci, cj, :] = (0, 0, 180)
            cv2.imwrite('../output/im_seg_{}.png'.format(_i), im_seg)

        # im_box
        if args.get_box:
            image_fn = rexp['image'] + '.png'
            img_path = os.path.join(img_dir, image_fn)
            im = cv2.imread(img_path)

            if len(output_obj_idxs) == 1:
                box_list = [sce['obj_bbox'][str(objid+1)] for objid in output_obj_idxs]
                im_box = draw_img_with_box(im, box_list)
                cv2.imwrite('../output/im_box_{}.png'.format(_i), im_box)
            else:
                print('[warn] Cannot get detection figure. The referred objects are more than 1.')


if __name__=='__main__':
    get_figure(args)
