from datasets.change_detection import ChangeDetection
from models.model_zoo import get_model
from utils.options import Options
from utils.palette import color_map
from utils.metric import F1_score, Precision
from utils.metric import IOUandSek
from utils.utils import label_onehot

from utils.RECOloss import compute_reco_loss, EMA, compute_unsupervised_loss_U2PL, compute_qualified_pseudo_label
from utils.loss import OhemCrossEntropy2dTensor

import numpy as np
import os
from PIL import Image
import shutil
import torch
#import torchcontrib
from torch.nn import CrossEntropyLoss, BCELoss, DataParallel
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import math
import copy

from tensorboardX import SummaryWriter



TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
train_log_dir = '/outdir/visualization' + TIMESTAMP
os.makedirs(train_log_dir, exist_ok=True)


class Trainer:
    def __init__(self, args):
        self.args = args

        trainset = ChangeDetection(root=args.data_root, mode="train")
        valset = ChangeDetection(root=args.data_root, mode="val")
        self.trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                      pin_memory=False, num_workers=16, drop_last=True)
        self.valloader = DataLoader(valset, batch_size=args.val_batch_size, shuffle=False,
                                    pin_memory=True, num_workers=16, drop_last=False)

        self.model = get_model(args.model, args.backbone, args.pretrained,
                               len(trainset.CLASSES), args.lightweight)

        if args.pretrain_from:
            self.model.load_state_dict(torch.load(args.pretrain_from), strict=False)

        if args.load_from:
            self.model.load_state_dict(torch.load(args.load_from), strict=True)

        weight_sem = torch.FloatTensor([2, 1, 2, 2, 1, 1]).cuda()
        self.criterion_sem = CrossEntropyLoss(ignore_index=-1, weight=weight_sem)
        # weight_scd = torch.FloatTensor([1, 2, 2, 2, 2, 2, 2]).cuda()
        # self.criterion_scd = CrossEntropyLoss(weight=weight_scd)
        self.criterion_scd = OhemCrossEntropy2dTensor()


        self.optimizer = Adam([{"params": [param for name, param in self.model.named_parameters()
                                           if "backbone" in name], "lr": args.lr},
                               {"params": [param for name, param in self.model.named_parameters()
                                           if "backbone" not in name], "lr": args.lr * 10.0}],
                              lr=args.lr, weight_decay=args.weight_decay)


        self.model = DataParallel(self.model).cuda()

        self.iters = 0
        self.total_iters = len(self.trainloader) * args.epochs
        self.previous_best = 0.0
        self.writer = SummaryWriter(train_log_dir)
        self.ema = EMA(self.model, 0.99)  # Mean teacher model

        self.percent = np.array([50,50,50,50,50,50])
        self.percent_end = np.array([80,80,80,80,80,80])
        self.star_pseudo = 15
        self.precision = np.array([0,0,0,0,0,0])

        self.temp = 0.5
        self.num_queries = 50
        self.num_negatives = 256

    def training(self, epoch):
        tbar = tqdm(self.trainloader)
        self.model.train()
        self.ema.model.train()

        total_loss = 0.0
        total_loss_sem_l = 0.0
        total_loss_sem_u = 0.0
        total_loss_scd = 0.0
        total_loss_contra = 0.0
        metric_precision = Precision(num_classes=len(ChangeDetection.CLASSES))
        precision = [0,0,0,0,0,0]

        self.writer.add_scalars('train/learning_rate', {'backbone':self.optimizer.param_groups[0]["lr"],\
            'head':self.optimizer.param_groups[1]["lr"]}, epoch)

        for i, (img1, img2, mask1, mask2, mask_bin) in enumerate(tbar):

            img1, img2 = img1.cuda(), img2.cuda()
            mask1, mask2 = mask1.cuda(), mask2.cuda()
            mask_bin = mask_bin.cuda()

            if epoch >= (self.star_pseudo-1):
                # generate pseudo-labels
                with torch.no_grad():
                    
                    out1_scd_teacher, out2_scd_teacher, out1_sem_teacher, out2_sem_teacher, out1_sem_rep_teacher, out2_sem_rep_teacher = self.ema.model(img1, img2)

                    pseudo_logits1, pseudo_labels1 = torch.max(torch.softmax(out1_sem_teacher, dim=1), dim=1)
                    pseudo_labels_u1 = pseudo_labels1 + 1
                    pseudo_labels_u1[mask_bin==1] = 0
                    pseudo_labels_u1 = pseudo_labels_u1.type(torch.int64)
                    pseudo_labels_l1 = pseudo_labels1 + 1
                    pseudo_labels_l1[mask_bin==0] = 0        
                    pseudo_labels_l1 = pseudo_labels_l1.type(torch.int64)         
                    pseudo_logits1[mask_bin==1] = 1

                    pseudo_logits2, pseudo_labels2 = torch.max(torch.softmax(out2_sem_teacher, dim=1), dim=1)
                    pseudo_labels_u2 = pseudo_labels2 + 1
                    pseudo_labels_u2[mask_bin==1] = 0
                    pseudo_labels_u2 = pseudo_labels_u2.type(torch.int64)
                    pseudo_labels_l2 = pseudo_labels2 + 1
                    pseudo_labels_l2[mask_bin==0] = 0  
                    pseudo_labels_l2 = pseudo_labels_l2.type(torch.int64)
                    pseudo_logits2[mask_bin==1] = 1

                    metric_precision.add_batch_mask(pseudo_labels_l1.cpu().numpy(), mask1.cpu().numpy())
                    metric_precision.add_batch_mask(pseudo_labels_l2.cpu().numpy(), mask2.cpu().numpy())
                    p1, p2, p3, p4, p5, p6 = metric_precision.evaluate_mask()
                    precision = [p1, p2, p3, p4, p5, p6]  


            out1_scd, out2_scd, out1_sem, out2_sem, out1_sem_rep, out2_sem_rep = self.model(img1, img2)

            loss_sem1_l = self.criterion_sem(out1_sem, mask1-1)
            loss_sem2_l = self.criterion_sem(out2_sem, mask2-1)        
            loss_scd1 = self.criterion_scd(out1_scd, mask1)
            loss_scd2 = self.criterion_scd(out2_scd, mask2)

            loss_sem_l = loss_sem1_l + loss_sem2_l
            loss_scd = loss_scd1 + loss_scd2


            if epoch >= self.star_pseudo:

                percent = self.percent + (self.percent_end-self.percent) * ((epoch-self.star_pseudo) / (29-self.star_pseudo))
                percent = percent*self.precision

                qualified_pseudo_labels_u1, weight1 = compute_qualified_pseudo_label(pseudo_labels_u1, percent, out1_sem_teacher)
                qualified_pseudo_labels_u1 = qualified_pseudo_labels_u1.type(torch.int64)
                qualified_pseudo_labels_u2, weight2 = compute_qualified_pseudo_label(pseudo_labels_u2, percent, out2_sem_teacher)
                qualified_pseudo_labels_u2 = qualified_pseudo_labels_u2.type(torch.int64)

                loss_sem1_u = compute_unsupervised_loss_U2PL(out1_sem, qualified_pseudo_labels_u1, weight1)
                loss_sem2_u = compute_unsupervised_loss_U2PL(out2_sem, qualified_pseudo_labels_u2, weight2)        

                loss_sem_u = loss_sem1_u + loss_sem2_u

                total_loss_sem_u += loss_sem_u.item() 

                with torch.no_grad():
                    
                    label1 = copy.deepcopy(mask1)
                    label1 += qualified_pseudo_labels_u1
                    label1 = label_onehot(label1, 7)[:,1:,:,:]
                    label1 = F.interpolate(label1, size=(64, 64), mode='bilinear', align_corners=False)
                    label1 = label1.type(torch.int64)
                    label2 = copy.deepcopy(mask2)
                    label2 += qualified_pseudo_labels_u2  
                    label2 = label_onehot(label2, 7)[:,1:,:,:]
                    label2 = F.interpolate(label2, size=(64, 64), mode='bilinear', align_corners=False)
                    label2 = label2.type(torch.int64)

                    prob1= torch.softmax(out1_sem, dim=1)
                    prob2= torch.softmax(out2_sem, dim=1)
                    prob1 = F.interpolate(prob1, size=(64, 64), mode='bilinear', align_corners=False)
                    prob2 = F.interpolate(prob2, size=(64, 64), mode='bilinear', align_corners=False)

            
                reco_loss1 = compute_reco_loss(out1_sem_rep, label1, prob1, self.temp, self.num_queries, self.num_negatives)
                reco_loss2 = compute_reco_loss(out2_sem_rep, label2, prob2, self.temp, self.num_queries, self.num_negatives)

                loss_contra = reco_loss1 + reco_loss2
                total_loss_contra += loss_contra.item() 

                loss = loss_sem_l + 2*loss_scd + 5*loss_sem_u + 0.2*loss_contra
                total_loss += loss.item()

            else:

                with torch.no_grad():
                    
                    label1 = copy.deepcopy(mask1)
                    label1 = label_onehot(label1, 7)[:,1:,:,:]
                    label1 = F.interpolate(label1, size=(64, 64), mode='bilinear', align_corners=False)
                    label1 = label1.type(torch.int64)
                    label2 = copy.deepcopy(mask2)  
                    label2 = label_onehot(label2, 7)[:,1:,:,:]
                    label2 = F.interpolate(label2, size=(64, 64), mode='bilinear', align_corners=False)
                    label2 = label2.type(torch.int64)

                    prob1= torch.softmax(out1_sem, dim=1)
                    prob2= torch.softmax(out2_sem, dim=1)
                    prob1 = F.interpolate(prob1, size=(64, 64), mode='bilinear', align_corners=False)
                    prob2 = F.interpolate(prob2, size=(64, 64), mode='bilinear', align_corners=False)

                reco_loss1 = compute_reco_loss(out1_sem_rep, label1, prob1, self.temp, self.num_queries, self.num_negatives)
                reco_loss2 = compute_reco_loss(out2_sem_rep, label2, prob2, self.temp, self.num_queries, self.num_negatives)

                loss_contra = reco_loss1 + reco_loss2
                total_loss_contra += loss_contra.item() 

                total_loss_sem_l += loss_sem_l.item()
                total_loss_scd += loss_scd.item()

                percent = [0,0,0,0,0,0]
                loss = loss_sem_l + 2*loss_scd + 0.2*loss_contra
                total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.ema.update(self.model)


            self.iters += 1
            lr = self.args.lr * (1 - self.iters / self.total_iters) ** 0.9
            self.optimizer.param_groups[0]["lr"] = lr
            self.optimizer.param_groups[1]["lr"] = lr * 10.0

            tbar.set_description("Loss: %.3f, Scd Loss: %.3f, Sem Loss label: %.3f, Sem Loss unlabel: %.3f, Contra Loss: %.10f, \
                            p1: %.3f, p2: %.3f, p3: %.3f, p4: %.3f, p5: %.3f, p6: %.3f" %
                                 (total_loss / (i + 1), total_loss_scd / (i + 1), total_loss_sem_l / (i + 1), total_loss_sem_u / (i + 1), total_loss_contra / (i + 1),\
                                percent[0], percent[1], percent[2], percent[3], percent[4], percent[5]))
            # break

        self.precision = precision

        self.writer.add_scalars('trainloss', {'loss_scd':total_loss_scd/len(self.trainloader),\
             'loss_sem_l':total_loss_sem_l/len(self.trainloader), 'loss_sem_u':total_loss_sem_u/len(self.trainloader), \
                'loss_contra':total_loss_contra/len(self.trainloader), 'total_loss':total_loss/len(self.trainloader)} , epoch)


    def validation(self, epoch):
        tbar = tqdm(self.valloader)
        self.model.eval()
        metric_sem = F1_score(num_classes=len(ChangeDetection.CLASSES))
        metric1_sem = IOUandSek(num_classes=len(ChangeDetection.CLASSES))
        metric_scd = F1_score(num_classes=len(ChangeDetection.CLASSES))
        metric1_scd = IOUandSek(num_classes=len(ChangeDetection.CLASSES))

        if self.args.save_mask:
            cmap = color_map()

        with torch.no_grad():
            for img1, img2, mask1, mask2, mask_bin, id in tbar:
                img1, img2 = img1.cuda(), img2.cuda()

                out1_scd, out2_scd, out1_sem, out2_sem, out1_sem_rep, out2_sem_rep = self.model(img1, img2)   
               
                out1_sem = torch.argmax(out1_sem, dim=1).cpu().numpy() + 1
                out2_sem = torch.argmax(out2_sem, dim=1).cpu().numpy() + 1        
                out1_scd = torch.argmax(out1_scd, dim=1).cpu().numpy()
                out2_scd = torch.argmax(out2_scd, dim=1).cpu().numpy()  
                out1_scd_bin = np.zeros_like(out1_scd)
                out1_scd_bin[out1_scd!=0] = 1
                out2_scd_bin = np.zeros_like(out2_scd)
                out2_scd_bin[out2_scd!=0] = 1
                out1_sem[out1_scd_bin == 0] = 0
                out2_sem[out2_scd_bin == 0] = 0

                metric_sem.add_batch_mask(out1_sem, mask1.numpy())
                metric_sem.add_batch_mask(out2_sem, mask2.numpy())

                f_0_sem, f_1_sem, f_2_sem, f_3_sem, f_4_sem, f_5_sem, f_6_sem = metric_sem.evaluate_mask()

                metric1_sem.add_batch(out1_sem, mask1.numpy())
                metric1_sem.add_batch(out2_sem, mask2.numpy())

                score_sem, miou_sem, sek_sem = metric1_sem.evaluate()


                metric_scd.add_batch_mask(out1_scd, mask1.numpy())
                metric_scd.add_batch_mask(out2_scd, mask2.numpy())
                metric_scd.add_batch_bin(out1_scd_bin, mask_bin.numpy())
                metric_scd.add_batch_bin(out2_scd_bin, mask_bin.numpy())

                f_0_scd, f_1_scd, f_2_scd, f_3_scd, f_4_scd, f_5_scd, f_6_scd = metric_scd.evaluate_mask()
                f_bin_scd = metric_scd.evaluate_bin()

                metric1_scd.add_batch(out1_scd, mask1.numpy())
                metric1_scd.add_batch(out2_scd, mask2.numpy())

                score_scd, miou_scd, sek_scd = metric1_scd.evaluate()


                tbar.set_description("score_scd: %.2f, miou_scd: %.2f, sek_scd: %.2f, f_0_scd: %.2f, f_1_scd: %.2f, f_2_scd: %.2f, f_3_scd: %.2f, f_4_scd: %.2f, f_5_scd: %.2f, f_6_scd: %.2f, f_bin_scd: %.2f,\
                score_sem: %.2f, miou_sem: %.2f, sek_sem: %.2f, f_0_sem: %.2f, f_1_sem: %.2f, f_2_sem: %.2f, f_3_sem: %.2f, f_4_sem: %.2f, f_5_sem: %.2f, f_6_sem: %.2f" % 
                    (score_scd * 100.0, miou_scd * 100.0, sek_scd * 100.0, f_0_scd * 100.0, f_1_scd * 100.0, f_2_scd * 100.0, f_3_scd * 100.0, f_4_scd * 100.0, f_5_scd * 100.0, f_6_scd * 100.0, f_bin_scd * 100.0,\
                     score_sem * 100.0, miou_sem * 100.0, sek_sem * 100.0, f_0_sem * 100.0, f_1_sem * 100.0, f_2_sem * 100.0, f_3_sem * 100.0, f_4_sem * 100.0, f_5_sem * 100.0, f_6_sem * 100.0))
                # break

        self.writer.add_scalar('val/f_0', f_0_scd, epoch)
        self.writer.add_scalar('val/f_1', f_1_scd, epoch)
        self.writer.add_scalar('val/f_2', f_2_scd, epoch)
        self.writer.add_scalar('val/f_3', f_3_scd, epoch)
        self.writer.add_scalar('val/f_4', f_4_scd, epoch)
        self.writer.add_scalar('val/f_5', f_5_scd, epoch)
        self.writer.add_scalar('val/f_6', f_6_scd, epoch)
        self.writer.add_scalar('val/f_bin', f_bin_scd, epoch)
        self.writer.add_scalar('val/score', score_scd, epoch)
        self.writer.add_scalar('val/miou', miou_scd, epoch)
        self.writer.add_scalar('val/sek', sek_scd, epoch)

        self.writer.add_scalar('val/f_0_seg', f_0_sem, epoch)
        self.writer.add_scalar('val/f_1_seg', f_1_sem, epoch)
        self.writer.add_scalar('val/f_2_seg', f_2_sem, epoch)
        self.writer.add_scalar('val/f_3_seg', f_3_sem, epoch)
        self.writer.add_scalar('val/f_4_seg', f_4_sem, epoch)
        self.writer.add_scalar('val/f_5_seg', f_5_sem, epoch)
        self.writer.add_scalar('val/f_6_seg', f_6_sem, epoch)
        self.writer.add_scalar('val/score_seg', score_sem, epoch)
        self.writer.add_scalar('val/miou_seg', miou_sem, epoch)
        self.writer.add_scalar('val/sek_seg', sek_sem, epoch)

             
        if self.args.load_from:
            exit(0)

        score_scd *= 100.0     
        dir_model = "outdir/models"
        os.makedirs(dir_model, exist_ok=True)
        if score_scd >= self.previous_best:
            if self.previous_best != 0:
                model_path = "outdir/models/%s_%s_%.2f.pth" % \
                             (self.args.model, self.args.backbone, self.previous_best)
                if os.path.exists(model_path):
                    os.remove(model_path)

            torch.save(self.model.module.state_dict(), "outdir/models/%s_%s_%.2f.pth" %
                       (self.args.model, self.args.backbone, score_scd))
            self.previous_best = score_scd


if __name__ == "__main__":
    args = Options().parse()
    trainer = Trainer(args)

    if args.load_from:
        trainer.validation()

    for epoch in range(args.epochs):
        print("\n==> Epoches %i, backbone learning rate = %.5f, head learning rate = %.5f\t\t\t\t previous best = %.2f" %
              (epoch, trainer.optimizer.param_groups[0]["lr"], trainer.optimizer.param_groups[1]["lr"], trainer.previous_best))
        trainer.training(epoch)
        trainer.validation(epoch)