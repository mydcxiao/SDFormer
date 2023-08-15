from .segmenter import CGFormer
from loguru import logger
import torch.nn as nn
from .backbone import MultiModalSwinTransformer
from .ldm_encoder import FeatureExtractorBackbone, LdmImplicitCaptionerExtractor
import torch


def build_model(args):
   # initialize the SwinTransformer backbone with the specified version
    if args.backbone == 'swin':
        if args.swin_type == 'tiny':
            embed_dim = 96
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
        elif args.swin_type == 'small':
            embed_dim = 96
            depths = [2, 2, 18, 2]
            num_heads = [3, 6, 12, 24]
        elif args.swin_type == 'base':
            embed_dim = 128
            depths = [2, 2, 18, 2]
            num_heads = [4, 8, 16, 32]
        elif args.swin_type == 'large':
            embed_dim = 192
            depths = [2, 2, 18, 2]
            num_heads = [6, 12, 24, 48]
        else:
            assert False
        # args.window12 added for test.py because state_dict is loaded after model initialization
        if 'window12' in args.swin_pretrain or args.window12:
            logger.info('Window size 12!')
            window_size = 12
        else:
            window_size = 7

        if args.mha:
            mha = args.mha.split('-')  # if non-empty, then ['a', 'b', 'c', 'd']
            mha = [int(a) for a in mha]
        else:
            mha = [1, 1, 1, 1]

        out_indices = (0, 1, 2, 3)
        backbone = MultiModalSwinTransformer(embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                                            window_size=window_size,
                                            ape=False, drop_path_rate=0.3, patch_norm=True,
                                            out_indices=out_indices,
                                            use_checkpoint=False, num_heads_fusion=mha,
                                            fusion_drop=args.fusion_drop
                                            )
        if args.swin_pretrain:
            logger.info('Initializing Multi-modal Swin Transformer weights from ' + args.swin_pretrain)
            backbone.init_weights(pretrained=args.swin_pretrain)
        else:
            logger.info('Randomly initialize Multi-modal Swin Transformer weights.')
            backbone.init_weights()
    else:
        backbone = FeatureExtractorBackbone(
            feature_extractor=LdmImplicitCaptionerExtractor(
                encoder_block_indices=(5, 7),
                unet_block_indices=(2, 5, 8, 11),
                decoder_block_indices=(2, 5),
                steps=(0,),
                learnable_time_embed=True,
                num_timesteps=1,
                clip_model_name="ViT-L-14-336",
            ),
            out_features=["s2", "s3", "s4", "s5"],
            use_checkpoint=True,
            slide_training=True,
        )
        if args.ldm_pretrain:
            logger.info('Initializing LDM encoder weights from ' + args.ldm_pretrain)
            with open(args.ldm_pretrain, "rb") as f:
                state_dict = torch.load(f)
            backbone.load_state_dict(state_dict, strict=False)
        else:
            logger.info('Randomly initialize LDM encoder weights.')

    model = CGFormer(backbone, args)
    
    return model


def build_segmenter(args, DDP=True, OPEN=False):
    model = build_model(args)
    if DDP:
        if args.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model.cuda(),
                                                    device_ids=[args.gpu],
                                                    find_unused_parameters=True
                                                    )

        single_model = model.module
        if OPEN:
            for p in single_model.backbone.parameters():
                p.requires_grad_(False)
        if args.backbone == 'swin':
            param_list = [
                {
                    "params": [
                        p
                        for n, p in single_model.named_parameters()
                        if "backbone" not in n and "text_encoder" not in n and p.requires_grad
                    ],

                },
                {
                    "params": [
                        p
                        for n, p in single_model.named_parameters()
                        if "pwam" in n and p.requires_grad
                    ],

                },
                {
                    "params": [p for n, p in single_model.named_parameters() if "backbone" in n and "pwam" not in n and p.requires_grad],
                    "lr": args.lr_backbone,
                },
                {
                    "params": [p for n, p in single_model.named_parameters() if "text_encoder" in n and p.requires_grad],
                    "lr": args.lr_text_encoder,
                },
            ]
        else:
            param_list = [
                {
                    "params": [
                        p
                        for n, p in single_model.named_parameters()
                        if "backbone" not in n and "text_encoder" not in n and p.requires_grad
                    ],

                },
                {
                    "params": 
                    [p for n, p in single_model.named_parameters() if "backbone" in n 
                    and "backbone.feature_extractor.ldm_extractor" not in n
                    and p.requires_grad
                    ],
                    "lr": args.lr_backbone,
                },
                {
                    "params": [p for n, p in single_model.named_parameters() if "text_encoder" in n and p.requires_grad],
                    "lr": args.lr_text_encoder,
                },
            ]

        return model, param_list
    else:
        model = nn.DataParallel(model).cuda()
        return model
