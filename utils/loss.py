import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable

def nll_loss(output, target):
    return F.nll_loss(output, target)

def bce_with_logits_loss():
    loss = torch.nn.BCEWithLogitsLoss(reduction='none')
    # print(output.size(),target.size())
    return loss

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations
    https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py
    '''


    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        # return -self.loss.sum()
        return -self.loss



def bce_loss(output, target):
    loss = torch.nn.BCELoss(reduction='none')
    return loss(output, target)

def focal_loss(outputs, targets, alpha=1, gamma=2, logits=True, reduce=True):
    if logits:
        BCE_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduce=False)
    else:
        BCE_loss = F.binary_cross_entropy(outputs, targets, reduce=False)
    pt = torch.exp(-BCE_loss)
    F_loss = alpha * (1 - pt) ** gamma * BCE_loss

    if reduce:
        return torch.mean(F_loss)
    else:
        return F_loss

def dice_loss(output, target, logits=True):
    if logits:
        output = nn.Sigmoid()(output)
    N = target.size(0)
    smooth = 1

    input_flat = output.view(N, -1)
    target_flat = target.view(N, -1)

    intersection = input_flat * target_flat

    loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)

    # print(intersection.sum(1))
    # print(input_flat.sum(1))
    # print(target_flat.sum(1))
    # print(loss)

    loss = 1 - loss.sum() / N

    return loss

def multiclass_dice_loss(output, target, weights=None, logits=True):
    C = target.shape[1]

    # if weights is None:
    # 	weights = torch.ones(C) #uniform weights for all classes

    totalLoss = 0

    for i in range(C):
        diceLoss = dice_loss(output[:, i], target[:, i], logits)
        # print(i, diceLoss)
        if weights is not None:
            diceLoss *= weights[i]
        totalLoss += diceLoss

    return totalLoss

## nn.Moudle
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, outputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(outputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target, logits=True):
        if logits:
            input = nn.Sigmoid()(input)
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss

class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None, logits=True):

        C = target.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        dice = DiceLoss()
        totalLoss = 0

        for i in range(C):
            diceLoss = dice(input[:, i], target[:, i], logits)
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss


class FocalLoss2d(nn.Module):
    # refered https://github.com/andrijdavid/FocalLoss/blob/master/focalloss.py

    def __init__(self, gamma=2, weight=None, size_average=True, ignore_index=-100, balance_param=0.25):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.weight = torch.from_numpy(weight).cuda()
        self.reduction = 'none'
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = balance_param

    def forward(self, output, target):
        # inputs and targets are assumed to be BatchxClasses
        assert len(output.shape) == len(target.shape)
        assert output.size(0) == target.size(0)
        assert output.size(1) == target.size(1)

        # weight = Variable(self.weight)

        # compute the negative likelyhood
        logpt = - F.binary_cross_entropy_with_logits(output, target, pos_weight=self.weight, reduction=self.reduction)
        # print(logpt.shape)
        pt = torch.exp(logpt)

        # compute the loss
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        if self.size_average:
            focal_loss = focal_loss.mean()
        else:
            focal_loss = focal_loss.sum()
        balanced_focal_loss = self.balance_param * focal_loss
        return balanced_focal_loss

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss
def linear_combination(x, y, epsilon):
    return epsilon*x + (1-epsilon)*y
class LabelSmoothingBCE(nn.Module):
    def __init__(self, epsilon:float=0.1, reduction='mean'):
        super(LabelSmoothingBCE, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, pred, target):
        n = pred.size()[-1]
        log_preds = F.logsigmoid(pred)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        bce = F.binary_cross_entropy(log_preds, target, reduction=self.reduction)
        return linear_combination(loss/n, bce, self.epsilon)
def label_smoothing_bce(output, target):
    loss = LabelSmoothingBCE()
    return loss(output, target)

class CustomBCE():
    def __init__(self):
        super(CustomBCE)
    def __call__(self, output, target, *args, **kwargs):
        output = torch.sigmoid(output)
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)
        loss = torch.neg(loss)
        return loss

# from utils.dataset import get_weights
def weighted_bce_with_logits_loss(output, target, weights):
    loss = F.binary_cross_entropy_with_logits(input=output, target=target, reduce=False)
    loss = torch.mm(loss, weights)
    loss = torch.mean(loss)
    # print('using weighted bce loss')
    # loss = torch.nn.BCEWithLogitsLoss()
    # print(output.size(),target.size())
    return loss

def custom_bce(output, target):
    output = torch.sigmoid(output)
    loss = CustomBCE()
    # print(output.size(),target.size())
    return loss(output, target)


class YoloLoss(nn.Module):
    # refer to https://github.com/abeardear/pytorch-YOLO-v1/blob/master/yoloLoss.py
    def __init__(self,feature_length,l_coord,l_noobj):
        super(YoloLoss,self).__init__()
        self.N = feature_length
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def forward(self, pred_tensor, target_tensor: torch.Tensor):
        # shape: (B, N, 3)  [c, x, type_num, peak_type]
        #
        #

        # deal with samples without peaks detected
        target_tmp = target_tensor.view(len(target_tensor), -1)
        target_max = target_tmp.max(1)[0]
        target_index = target_max.gt(0.5)
        target_tensor = target_tensor[target_index]
        pred_tensor = pred_tensor[target_index]

        coo_mask = target_tensor[:, :, :, 0] > 0.5 #>0
        noo_mask = target_tensor[:, :, :, 0] < 0.5 #== 0
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)

        coo_pred = pred_tensor[coo_mask].view(-1, 3)
        coo_target = target_tensor[coo_mask].view(-1, 3)

        # when no object
        # optimize confidence
        noo_pred = pred_tensor[noo_mask].view(-1, 3)
        noo_target = target_tensor[noo_mask].view(-1, 3)
        noo_pred_mask = torch.zeros_like(noo_pred)
        noo_pred_mask.zero_()
        noo_pred_mask[:, 0] = 1
        noo_pred_mask = noo_pred_mask.bool()
        noo_pred_c = noo_pred[noo_pred_mask]  # noo pred只需要计算 c 的损失 size[-1,2]
        noo_target_c = noo_target[noo_pred_mask]
        no_obj_loss = F.mse_loss(noo_pred_c, noo_target_c, reduction='sum')

        # when there is a object
        c_pred = coo_pred[:, :1]
        x_pred = coo_pred[:, 1:2]
        class_pred = coo_pred[:, 2:]
        c_target = coo_target[:, :1]
        x_target = coo_target[:, 1:2]
        class_target = coo_target[:, 2:]


        c_loss = F.mse_loss(c_pred, c_target, reduction='sum')

        loc_loss = F.mse_loss(x_pred, x_target, reduction='sum')

        class_loss = F.mse_loss(class_pred, class_target, reduction='sum')

        return (self.l_noobj * no_obj_loss + c_loss + class_loss + self.l_coord * loc_loss) / (self.N * len(target_tensor))

if __name__ == "__main__":
    a = torch.rand((16, 24))
    b = torch.rand((16, 24))
    w = torch.rand((24, 1))
    loss = weighted_bce_with_logits_loss(a, b, w)
    print(loss)
    print('test')