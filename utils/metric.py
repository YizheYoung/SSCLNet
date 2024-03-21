#-------------SECOND---------------------------
import math
import numpy as np


class F1_score:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist_mask = np.zeros((num_classes, num_classes))
        self.hist_bin = np.zeros((2, 2))


    def _fast_hist_mask(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist_mask = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist_mask

    def _fast_hist_bin(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < 2)
        hist_bin = np.bincount(
            2 * label_true[mask].astype(int) +
            label_pred[mask], minlength=4).reshape(2, 2)
        return hist_bin

    def add_batch_mask(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist_mask += self._fast_hist_mask(lp.flatten(), lt.flatten())

    def add_batch_bin(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist_bin += self._fast_hist_bin(lp.flatten(), lt.flatten())        


    def evaluate_mask(self):
        precision_0 = self.hist_mask[0][0]/(self.hist_mask[1][0] + self.hist_mask[0][0] + self.hist_mask[2][0] + self.hist_mask[3][0] + self.hist_mask[4][0] + self.hist_mask[5][0] + self.hist_mask[6][0])
        recall_0 = self.hist_mask[0][0]/(self.hist_mask[0][1]+ self.hist_mask[0][0] + self.hist_mask[0][2] + self.hist_mask[0][3] + self.hist_mask[0][4] + self.hist_mask[0][5] + self.hist_mask[0][6])
        f_0 = 2*precision_0*recall_0/(precision_0 + recall_0)

        precision_1 = self.hist_mask[1][1]/(self.hist_mask[1][1] + self.hist_mask[0][1] + self.hist_mask[2][1] + self.hist_mask[3][1] + self.hist_mask[4][1] + self.hist_mask[5][1] + self.hist_mask[6][1])
        recall_1 = self.hist_mask[1][1]/(self.hist_mask[1][1]+ self.hist_mask[1][0] + self.hist_mask[1][2] + self.hist_mask[1][3] + self.hist_mask[1][4] + self.hist_mask[1][5] + self.hist_mask[1][6])
        f_1 = 2*precision_1*recall_1/(precision_1 + recall_1)

        precision_2 = self.hist_mask[2][2]/(self.hist_mask[2][2] + self.hist_mask[0][2] + self.hist_mask[1][2] + self.hist_mask[3][2] + self.hist_mask[4][2] + self.hist_mask[5][2] + self.hist_mask[6][2])
        recall_2 = self.hist_mask[2][2]/(self.hist_mask[2][2]+ self.hist_mask[2][1] + self.hist_mask[2][0] + self.hist_mask[2][3] + self.hist_mask[2][4] + self.hist_mask[2][5] + self.hist_mask[2][6])
        f_2 = 2*precision_2*recall_2/(precision_2 + recall_2)

        precision_3 = self.hist_mask[3][3]/(self.hist_mask[3][3] + self.hist_mask[0][3] + self.hist_mask[1][3] + self.hist_mask[2][3] + self.hist_mask[4][3] + self.hist_mask[5][3] + self.hist_mask[6][3])
        recall_3 = self.hist_mask[3][3]/(self.hist_mask[3][3]+ self.hist_mask[3][1] + self.hist_mask[3][2] + self.hist_mask[3][0] + self.hist_mask[3][4] + self.hist_mask[3][5] + self.hist_mask[3][6])
        f_3 = 2*precision_3*recall_3/(precision_3 + recall_3)
        
        precision_4 = self.hist_mask[4][4]/(self.hist_mask[4][4] + self.hist_mask[0][4] + self.hist_mask[1][4] + self.hist_mask[2][4] + self.hist_mask[3][4] + self.hist_mask[5][4] + self.hist_mask[6][4])
        recall_4 = self.hist_mask[4][4]/(self.hist_mask[4][4]+ self.hist_mask[4][1] + self.hist_mask[4][2] + self.hist_mask[4][3] + self.hist_mask[4][0] + self.hist_mask[4][5] + self.hist_mask[4][6])
        f_4 = 2*precision_4*recall_4/(precision_4 + recall_4)

        precision_5 = self.hist_mask[5][5]/(self.hist_mask[5][5] + self.hist_mask[0][5] + self.hist_mask[1][5] + self.hist_mask[2][5] + self.hist_mask[3][5] + self.hist_mask[4][5] + self.hist_mask[6][5])
        recall_5 = self.hist_mask[5][5]/(self.hist_mask[5][5]+ self.hist_mask[5][1] + self.hist_mask[5][2] + self.hist_mask[5][3] + self.hist_mask[5][0] + self.hist_mask[5][4] + self.hist_mask[5][6])
        f_5 = 2*precision_5*recall_5/(precision_5 + recall_5)

        precision_6 = self.hist_mask[6][6]/(self.hist_mask[6][6] + self.hist_mask[0][6] + self.hist_mask[1][6] + self.hist_mask[2][6] + self.hist_mask[3][6] + self.hist_mask[4][6] + self.hist_mask[5][6])
        recall_6 = self.hist_mask[6][6]/(self.hist_mask[6][6]+ self.hist_mask[6][1] + self.hist_mask[6][2] + self.hist_mask[6][3] + self.hist_mask[6][0] + self.hist_mask[6][4] + self.hist_mask[6][5])
        f_6 = 2*precision_6*recall_6/(precision_6 + recall_6)

        return f_0, f_1, f_2, f_3, f_4, f_5, f_6


    def evaluate_bin(self):
        precision_bin = self.hist_bin[1][1]/(self.hist_bin[1][1] + self.hist_bin[1][0])
        recall_bin = self.hist_bin[1][1]/(self.hist_bin[1][1]+ self.hist_bin[0][1])
        f_bin = 2*precision_bin*recall_bin/(precision_bin + recall_bin)

        return f_bin

    def confusion_matrix(self):
        return self.hist_mask

class Precision:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist_mask = np.zeros((num_classes, num_classes))

    def _fast_hist_mask(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist_mask = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist_mask

    def add_batch_mask(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist_mask += self._fast_hist_mask(lp.flatten(), lt.flatten())


    def evaluate_mask(self):

        precision_1 = self.hist_mask[1][1]/(self.hist_mask[1][1] + self.hist_mask[0][1] + self.hist_mask[2][1] + self.hist_mask[3][1] + self.hist_mask[4][1] + self.hist_mask[5][1] + self.hist_mask[6][1])
        recall_1 = self.hist_mask[1][1]/(self.hist_mask[1][1]+ self.hist_mask[1][0] + self.hist_mask[1][2] + self.hist_mask[1][3] + self.hist_mask[1][4] + self.hist_mask[1][5] + self.hist_mask[1][6])
        f_1 = 2*precision_1*recall_1/(precision_1 + recall_1)

        precision_2 = self.hist_mask[2][2]/(self.hist_mask[2][2] + self.hist_mask[0][2] + self.hist_mask[1][2] + self.hist_mask[3][2] + self.hist_mask[4][2] + self.hist_mask[5][2] + self.hist_mask[6][2])
        recall_2 = self.hist_mask[2][2]/(self.hist_mask[2][2]+ self.hist_mask[2][1] + self.hist_mask[2][0] + self.hist_mask[2][3] + self.hist_mask[2][4] + self.hist_mask[2][5] + self.hist_mask[2][6])
        f_2 = 2*precision_2*recall_2/(precision_2 + recall_2)

        precision_3 = self.hist_mask[3][3]/(self.hist_mask[3][3] + self.hist_mask[0][3] + self.hist_mask[1][3] + self.hist_mask[2][3] + self.hist_mask[4][3] + self.hist_mask[5][3] + self.hist_mask[6][3])
        recall_3 = self.hist_mask[3][3]/(self.hist_mask[3][3]+ self.hist_mask[3][1] + self.hist_mask[3][2] + self.hist_mask[3][0] + self.hist_mask[3][4] + self.hist_mask[3][5] + self.hist_mask[3][6])
        f_3 = 2*precision_3*recall_3/(precision_3 + recall_3)
        
        precision_4 = self.hist_mask[4][4]/(self.hist_mask[4][4] + self.hist_mask[0][4] + self.hist_mask[1][4] + self.hist_mask[2][4] + self.hist_mask[3][4] + self.hist_mask[5][4] + self.hist_mask[6][4])
        recall_4 = self.hist_mask[4][4]/(self.hist_mask[4][4]+ self.hist_mask[4][1] + self.hist_mask[4][2] + self.hist_mask[4][3] + self.hist_mask[4][0] + self.hist_mask[4][5] + self.hist_mask[4][6])
        f_4 = 2*precision_4*recall_4/(precision_4 + recall_4)

        precision_5 = self.hist_mask[5][5]/(self.hist_mask[5][5] + self.hist_mask[0][5] + self.hist_mask[1][5] + self.hist_mask[2][5] + self.hist_mask[3][5] + self.hist_mask[4][5] + self.hist_mask[6][5])
        recall_5 = self.hist_mask[5][5]/(self.hist_mask[5][5]+ self.hist_mask[5][1] + self.hist_mask[5][2] + self.hist_mask[5][3] + self.hist_mask[5][0] + self.hist_mask[5][4] + self.hist_mask[5][6])
        f_5 = 2*precision_5*recall_5/(precision_5 + recall_5)

        precision_6 = self.hist_mask[6][6]/(self.hist_mask[6][6] + self.hist_mask[0][6] + self.hist_mask[1][6] + self.hist_mask[2][6] + self.hist_mask[3][6] + self.hist_mask[4][6] + self.hist_mask[5][6])
        recall_6 = self.hist_mask[6][6]/(self.hist_mask[6][6]+ self.hist_mask[6][1] + self.hist_mask[6][2] + self.hist_mask[6][3] + self.hist_mask[6][0] + self.hist_mask[6][4] + self.hist_mask[6][5])
        f_6 = 2*precision_6*recall_6/(precision_6 + recall_6)
        
        return f_1, f_2, f_3, f_4, f_5, f_6



def cal_kappa(hist):
    if hist.sum() == 0:
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
    return kappa

class IOUandSek:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        confusion_matrix = np.zeros((2, 2))
        confusion_matrix[0][0] = self.hist[0][0]
        confusion_matrix[0][1] = self.hist.sum(1)[0] - self.hist[0][0]
        confusion_matrix[1][0] = self.hist.sum(0)[0] - self.hist[0][0]
        confusion_matrix[1][1] = self.hist[1:, 1:].sum()

        iou = np.diag(confusion_matrix) / (confusion_matrix.sum(0) +
                                           confusion_matrix.sum(1) - np.diag(confusion_matrix))
        miou = np.mean(iou)

        hist = self.hist.copy()
        hist[0][0] = 0
        kappa = cal_kappa(hist)
        sek = kappa * math.exp(iou[1] - 1)

        score = 0.3 * miou + 0.7 * sek

        return score, miou, sek

    # def miou(self):
    #     confusion_matrix = self.hist[1:, 1:]
    #     iou = np.diag(confusion_matrix) / (confusion_matrix.sum(0) + confusion_matrix.sum(1) - np.diag(confusion_matrix))
    #     return iou, np.mean(iou)



