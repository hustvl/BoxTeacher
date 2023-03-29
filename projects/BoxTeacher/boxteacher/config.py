from detectron2.config import CfgNode as CN


def add_box_teacher_config(cfg):
    
    
    cfg.MODEL.BOX_TEACHER = CN()
    
    # filter and assignment
    cfg.MODEL.BOX_TEACHER.IOU_THR = 0.5
    cfg.MODEL.BOX_TEACHER.SCORE_THR = 0.0
    # EMA
    cfg.MODEL.BOX_TEACHER.MOMENTUM = 0.999
    
    # loss for pseudo masks
    # mask weight for pseudo loss
    cfg.MODEL.BOX_TEACHER.MASK_WEIGHT = 0.5
    # warmup for pseudo loss
    cfg.MODEL.BOX_TEACHER.WITH_WARMUP = True
    cfg.MODEL.BOX_TEACHER.WARMUP_ITERS = 10000
    cfg.MODEL.BOX_TEACHER.WARMUP_METHOD = "linear"
    # add avg projection loss
    cfg.MODEL.BOX_TEACHER.WITH_AVG_LOSS = False
    cfg.MODEL.BOX_TEACHER.AVG_LOSS_WEIGHT = 0.1
    # affinity loss
    cfg.MODEL.BOX_TEACHER.MASK_AFFINITY_THRESH = 0.5
    cfg.MODEL.BOX_TEACHER.MASK_AFFINITY_WEIGHT = 0.1
    # fix reduction factor for pseudo loss
    cfg.MODEL.BOX_TEACHER.FIX_REDUCTION = True

    # inference using teacher
    cfg.MODEL.BOX_TEACHER.USE_TEACHER_INFERENCE = True
    # teacher with dynamic batch norm
    cfg.MODEL.BOX_TEACHER.TEACHER_EVAL = False

    # using augmentation
    cfg.MODEL.BOX_TEACHER.USE_AUG = False
    
    # return float masks instead of binary masks
    cfg.MODEL.BOX_TEACHER.RETURN_FLOAT_MASK = False
    # mask threshold for teacher
    cfg.MODEL.BOX_TEACHER.TEACHER_MASK_THRESHOLD = 0.5
    
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0
    cfg.SOLVER.OPTIMIZER = "SGD"


    # augmentation strong, weak, none
    cfg.INPUT.AUG_TYPE = "strong"
    cfg.INPUT.AUG_EXTRA = True

    # Swin Transformer
    cfg.MODEL.SWIN_TRANSFORMER = CN()
    cfg.MODEL.SWIN_TRANSFORMER.EMBED_DIM = 96
    cfg.MODEL.SWIN_TRANSFORMER.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]
    cfg.MODEL.SWIN_TRANSFORMER.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN_TRANSFORMER.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN_TRANSFORMER.WINDOW_SIZE = 7
    cfg.MODEL.SWIN_TRANSFORMER.MLP_RATIO = 4
    cfg.MODEL.SWIN_TRANSFORMER.DROP_PATH_RATE = 0.2
    cfg.MODEL.SWIN_TRANSFORMER.APE = False
