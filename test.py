import argparse
import joblib,copy
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch,sys
from tqdm import tqdm

from collections import OrderedDict
from config_test import parse_args
from lib.visualize import save_img,group_images,concat_result, single_result, single_result_prob
import os
from lib.logger import Logger, Print_Logger
from lib.extract_patches import *
from os.path import join
from lib.dataset import TestDataset
from lib.metrics import Evaluate
import models
from lib.common import setpu_seed,dict_round
from lib.pre_processing import my_PreProc


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


setpu_seed(2021)

class Test():
    def __init__(self, args):
        self.args = args
        assert (args.stride_height <= args.test_patch_height and args.stride_width <= args.test_patch_width)
        # save path
        self.path_experiment = join(args.outf, args.save)

        self.patches_imgs_test, self.test_imgs, self.test_masks, self.test_FOVs, self.new_height, self.new_width = get_data_test_overlap(
            test_data_path_list=args.test_data_path_list,
            patch_height=args.test_patch_height,
            patch_width=args.test_patch_width,
            stride_height=args.stride_height,
            stride_width=args.stride_width
        )
        self.img_height = self.test_imgs.shape[2]
        self.img_width = self.test_imgs.shape[3]

        test_set = TestDataset(self.patches_imgs_test)
        self.test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Inference prediction process
    def inference(self, net):
        net.eval()
        preds = []
        with torch.no_grad():
            for batch_idx, inputs in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                inputs = inputs.cuda()
                outputs = net(inputs)
                outputs = outputs[:,1].data.cpu().numpy()
                preds.append(outputs)
        predictions = np.concatenate(preds, axis=0)
        self.pred_patches = np.expand_dims(predictions,axis=1)
        
    # Evaluate ate and visualize the predicted images
    def evaluate(self):
        self.pred_imgs = recompone_overlap(
            self.pred_patches, self.new_height, self.new_width, self.args.stride_height, self.args.stride_width)
        ## restore to original dimensions
        self.pred_imgs = self.pred_imgs[:, :, 0:self.img_height, 0:self.img_width]

        #predictions only inside the FOV
        y_scores, y_true = pred_only_in_FOV(self.pred_imgs, self.test_masks, self.test_FOVs)
        eval = Evaluate(save_path=self.path_experiment)
        eval.add_batch(y_true, y_scores)
        log = eval.save_all_result(plot_curve=True,save_name="performance.txt")
        # save labels and probs for plot ROC and PR curve when k-fold Cross-validation
        np.save('{}/result.npy'.format(self.path_experiment), np.asarray([y_true, y_scores]))
        return dict_round(log, 6)

    # save segmentation imgs
    def save_segmentation_result(self):
        img_path_list, _, _ = load_file_path_txt(self.args.test_data_path_list)
        img_name_list = [item.split('/')[-1].split('.')[0] for item in img_path_list]

        kill_border(self.pred_imgs, self.test_FOVs) # only for visualization

        self.save_img_path = join(self.path_experiment,'result_img')
        # self.save_img_path = join(self.path_experiment,'drive_on_stare')
        # self.save_img_path = join(self.path_experiment,'chase_on_drive')
        # self.save_img_path = join(self.path_experiment,'stare_on_drive')
        # self.save_img_path = join(self.path_experiment,'result_prob')

        if not os.path.exists(join(self.save_img_path)):
            os.makedirs(self.save_img_path)
        # self.test_imgs = my_PreProc(self.test_imgs) # Uncomment to save the pre processed image
        for i in range(self.test_imgs.shape[0]):

            total_img = concat_result(self.test_imgs[i],self.pred_imgs[i],self.test_masks[i])    #All
            # total_img = single_result(self.test_imgs[i],self.pred_imgs[i],self.test_masks[i])     # binary
            # total_img = single_result_prob(self.test_imgs[i],self.pred_imgs[i],self.test_masks[i])     # prob


            save_img(total_img,join(self.save_img_path, "Result_"+img_name_list[i]+'.png'))

    # Val on the test set at each epoch
    def val(self):
        self.pred_imgs = recompone_overlap(
            self.pred_patches, self.new_height, self.new_width, self.args.stride_height, self.args.stride_width)
        ## recover to original dimensions
        self.pred_imgs = self.pred_imgs[:, :, 0:self.img_height, 0:self.img_width]

        #predictions only inside the FOV
        y_scores, y_true = pred_only_in_FOV(self.pred_imgs, self.test_masks, self.test_FOVs)
        eval = Evaluate(save_path=self.path_experiment)
        eval.add_batch(y_true, y_scores)
        confusion,accuracy,specificity,sensitivity,precision = eval.confusion_matrix()
        log = OrderedDict([('val_auc_roc', eval.auc_roc()),
                           ('val_f1', eval.f1_score()),
                           ('val_acc', accuracy),
                           ('SE', sensitivity),
                           ('SP', specificity)])
        return dict_round(log, 6)


if __name__ == '__main__':
    args = parse_args()
    # save_path = args.outf
    save_path = join(args.outf, args.save)
    sys.stdout = Print_Logger(os.path.join(save_path, 'test_log.txt'))
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # net = models.U_Net_DConv(1, 2).to(device)
    # net = models.U_Net_ODConv(1, 2).to(device)
    # net = models.U_Net_DR(1, 2).to(device)
    # net = models.U_Net_CondConv(1, 2).to(device)
    # net = models.U_Net_DeformConv(1,2).to(device)

    net = models.UNet_extract_part_PVF_DPR(1, 2).to(device)

    # net = models.UNetFamily.AttU_Net(1,2).to(device)
    # net = models.Self_att_gate_UNet.AttU_Net(1,2).to(device)
    # net = models.Dynamic_Gabor_UNet.U_Net(1,2).to(device)
    # net = models.unet_apf_DNL_CBAM.U_Net_apf_DNL(1,2).to(device)
    # net = models.LadderNet(1,2).to(device)
    # net = models.FR_UNet(2,1).to(device)
    # net = models.SegNet(1,2).to(device)
    # net = models.PSPNet(layers=18, bins=(1, 2, 3, 6), dropout=0.1, classes=2, use_ppm=True, pretrained=True).to(device)
    # net = models.DUNetV1V2(1, 2).to(device)
    # net = models.CC_Net(2).to(device)
    # net = models.Dynamic_Gabor_UNet.U_Net(1,2).to(device)
    # net = models.Iternet(1, 2).to(device)
    # net = models.UNetFamily.Dense_Unet(1,2).to(device)
    # net = models.SA_UNet(1, 2).to(device)
    # net = models.UNetFamily.U_Net(1,2).to(device)
    # net = models.UNetFamily.R2U_Net(1,2).to(device)
    # net = models.Self_att_gate_UNet.Att_map_U_Net_small(1,2).to(device)
    # net = models.OCENet(1,2).to(device)
    # net = models.OCENet_mine(1, 2).to(device)
    # net = models.SK_gabor_UNet.U_Net_all_fusion(1, 2).to(device)

    cudnn.benchmark = True

    # Load checkpoint
    print('==> Loading checkpoint...')
    checkpoint = torch.load(join(save_path, 'best_model.pth'))
    net.load_state_dict(checkpoint['net'])

    eval = Test(args)
    eval.inference(net)
    print(eval.evaluate())
    eval.save_segmentation_result()
