import torch
import torch.nn as nn
from .layers import Decoder
import torch.nn.functional as F
from bert.modeling_bert import BertModel

def dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """

    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()

def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """

    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean()

def build_attention_mask(context_length):
    # lazily create causal attention mask, with full attention between the vision tokens
    # pytorch uses additive attention mask; fill with -inf
    mask = torch.empty(context_length, context_length)
    mask.fill_(float("-inf"))
    mask.triu_(1)  # zero out the lower diagonal
    return mask

class CGFormer(nn.Module):
    def __init__(self, backbone, args):
        super(CGFormer, self).__init__()
        self.backbone = backbone
        self.decoder = Decoder(args)
        if args.backbone == 'swin':
            self.text_encoder = BertModel.from_pretrained(args.bert)
            self.text_encoder.pooler = None
        else:
            self.text_encoder = self.backbone.feature_extractor.clip.clip
        self.backbone_type = args.backbone

    def forward(self, x, text, l_mask, mask=None):
        input_shape = x.shape[-2:]
        if self.backbone_type == 'swin':
            l_feats = self.text_encoder(text, attention_mask=l_mask)[0]  # (6, 10, 768)
            l_feats = l_feats.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
            l_mask = l_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)
        else:
            l = self.text_encoder.token_embedding(text)  # [batch_size, n_ctx, d_model]
            l = l + self.text_encoder.positional_embedding[:l.size(1)]
            l = l.permute(1, 0, 2)  # NLD -> LND
            l = self.text_encoder.transformer(l, attn_mask=build_attention_mask(l.size(0)).to(l.device))
            l = l.permute(1, 0, 2)  # LND -> NLD
            l = self.text_encoder.ln_final(l)
            l_feats = l
            l_feats = l_feats.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
            l_mask = l_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)
            # text_embed = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_encoder.text_projection

        ##########################
        if self.backbone_type == 'ldm':
            features = self.backbone(x)
            x_c1, x_c2, x_c3, x_c4 = features['s2'], features['s3'], features['s4'], features['s5']
        else:
            features = self.backbone(x, l_feats, l_mask)
            x_c1, x_c2, x_c3, x_c4 = features
        pred, maps = self.decoder([x_c4, x_c3, x_c2, x_c1], l_feats, l_mask)
        pred = F.interpolate(pred, input_shape, mode='bilinear', align_corners=True)
        # loss
        if self.training:
            loss = 0.
            mask = mask.unsqueeze(1).float()
            for m, lam in zip(maps, [0.001,0.01,0.1]):
                m = m[:,1].unsqueeze(1)
                if m.shape[-2:] != mask.shape[-2:]:
                    mask_ = F.interpolate(mask, m.shape[-2:], mode='nearest').detach()
                loss += dice_loss(m, mask_) * lam
            loss += dice_loss(pred, mask) + sigmoid_focal_loss(pred, mask, alpha=-1, gamma=0)
            return pred.detach(), mask, loss
        else:
            return pred.detach(), maps
    
    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_type == 'ldm':
            self.backbone.feature_extractor.ldm_extractor.ldm.eval()
        return self