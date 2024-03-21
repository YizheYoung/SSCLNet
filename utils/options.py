import argparse

class Options:
    def __init__(self):
        parser = argparse.ArgumentParser('Contrastive-ST')
        parser.add_argument("--data-root", type=str, default="/data/SECOND") 
        parser.add_argument("--batch-size", type=int, default=8)#8
        parser.add_argument("--val-batch-size", type=int, default=16)
        parser.add_argument("--test-batch-size", type=int, default=16)
        parser.add_argument("--epochs", type=int, default=30)#30
        parser.add_argument("--lr", type=float, default=0.00015)#0.00015
        parser.add_argument("--weight-decay", type=float, default=1e-4)
        parser.add_argument("--backbone", type=str, default="hrnet_w40")
        parser.add_argument("--model", type=str, default="fcn")
        parser.add_argument("--lightweight", dest="lightweight", action="store_true",
                           help='lightweight head for fewer parameters and faster speed')
        parser.add_argument("--pretrain-from", type=str, help='train from a checkpoint')
        parser.add_argument("--load-from", type=str,
                           help='load trained model to generate predictions of validation set')
        parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                           help='initialize the backbone with pretrained parameters')
        parser.add_argument("--tta", dest="tta", action="store_true",
                           help='test-time augmentation')
        parser.add_argument("--save-mask", dest="save_mask", action="store_true",
                           help='save predictions of validation set during training')
        parser.add_argument("--use-pseudo-label", dest="use_pseudo_label", action="store_true",
                           help='use pseudo labels for re-training (must pseudo label first)')

        # parser.add_argument('--num_negatives', default=512, type=int, 
        #                     help='number of negative keys')
        # parser.add_argument('--num_queries', default=512, type=int, 
        #                     help='number of queries per segment per image')

        # parser.add_argument('--output_dim', default=256, type=int, 
        #                     help='output dimension from representation head')
        # parser.add_argument('--temp', default=0.5, type=float)
        # parser.add_argument('--strong_threshold', default=1.0, type=float, 
        #                     help='strong_threshold')


        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        print(args)
        return args
