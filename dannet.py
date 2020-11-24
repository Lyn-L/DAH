import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch

class DAN(nn.Module):

    def __init__(self, dim_feat, dim_latent=10, hash_bit=64, label=10):
        super(DAN, self).__init__()


        self.in_feature_1 = dim_feat
        self.out_feature_1 = dim_latent
        self.in_feature_2 = dim_latent
        self.out_feature_2 = dim_feat

        self.fc1_1 = nn.Linear(self.in_feature_1, self.out_feature_1)
        self.fc1_2 = nn.Linear(self.in_feature_2, self.out_feature_2)

        self.fc2_1 = nn.Linear(self.in_feature_1, self.out_feature_1)
        self.fc2_2 = nn.Linear(self.in_feature_2, self.out_feature_2)

        self.fc_to_cla = nn.Linear(self.out_feature_2, label)
        self.fc_to_lsh = nn.Linear(self.out_feature_2, hash_bit)

        self.dp = nn.Dropout(0.1)

        self.CrossEntropyLoss = nn.MultiLabelMarginLoss()
        self.MSELoss_lsh = nn.MSELoss()
        self.MSELoss_recon = nn.MSELoss()
        self.kl = nn.KLDivLoss(reduce=True)

    def hash(self, x):
        out_fc1_1 = F.leaky_relu(self.fc1_1(x),negative_slope=0.05)
        out_fc1_2 = F.leaky_relu(self.fc1_2(out_fc1_1),negative_slope=0.05)
        sum_1 = x + out_fc1_2

        out_fc2_1 = F.leaky_relu(self.fc2_1(sum_1),negative_slope=0.05)
        out_fc2_2 = F.leaky_relu(self.fc2_2(out_fc2_1),negative_slope=0.05)
        sum_2 = sum_1 + out_fc2_2 + x

        out = torch.tanh(self.fc_to_lsh(sum_2))

        return out

    def forward(self, x = None, target_data = None, lsh = None, labels = None):
        out_fc1_1 = F.leaky_relu(self.dp(self.fc1_1(x)),negative_slope=0.05)
        out_fc1_2 = F.leaky_relu(self.dp(self.fc1_2(out_fc1_1)),negative_slope=0.05)
        sum_1 = x + out_fc1_2

        out_fc2_1 = F.leaky_relu(self.dp(self.fc2_1(sum_1)),negative_slope=0.05)
        out_fc2_2 = F.leaky_relu(self.dp(self.fc2_2(out_fc2_1)),negative_slope=0.05)
        out = out_fc2_2 + out_fc1_2 + x

        out_fc_to_cla = self.fc_to_cla(out)
        out_fc_to_lsh = torch.tanh(self.fc_to_lsh(out))

        loss_regression = self.MSELoss_recon(out, target_data)
        loss_classification = self.CrossEntropyLoss(out_fc_to_cla, labels)
        loss_lsh = self.MSELoss_lsh(out_fc_to_lsh, lsh)
        loss_kl = self.kl(F.log_softmax(out_fc_to_lsh), F.softmax(lsh))
        loss =  loss_regression + 1.5 * loss_classification + loss_lsh - loss_kl

        return loss