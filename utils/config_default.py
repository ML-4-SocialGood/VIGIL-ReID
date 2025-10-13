from yacs.config import CfgNode as CN


def get_cfg_default():
    # ====================
    # Global CfgNode
    # ====================
    _C = CN()
    _C.OUTPUT_DIR = "./output/"
    _C.SEED = 41

    # ====================
    # Input CfgNode
    # ====================
    _C.INPUT = CN()
    _C.INPUT.SIZE = (224, 224)
    _C.INPUT.INTERPOLATION = "bilinear"
    _C.INPUT.TRANSFORMS = []
    _C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
    _C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
    # Transform Setting
    # Random Crop
    _C.INPUT.CROP_PADDING = 4
    # Random Resized Crop
    _C.INPUT.RRCROP_SCALE = (0.08, 1.0)
    # Cutout
    _C.INPUT.CUTOUT_N = 1
    _C.INPUT.CUTOUT_LEN = 16
    # Gaussian Noise
    _C.INPUT.GN_MEAN = 0.0
    _C.INPUT.GN_STD = 0.15
    # RandomAugment
    _C.INPUT.RANDAUGMENT_N = 2
    _C.INPUT.RANDAUGMENT_M = 10
    # ColorJitter (Brightness, Contrast, Saturation, Hue)
    _C.INPUT.COLORJITTER_B = 0.4
    _C.INPUT.COLORJITTER_C = 0.4
    _C.INPUT.COLORJITTER_S = 0.4
    _C.INPUT.COLORJITTER_H = 0.1
    # Random Gray Scale's Probability
    _C.INPUT.RGS_P = 0.2
    # Gaussian Blur
    _C.INPUT.GB_P = 0.5  # Probability of Applying Gaussian Blur
    _C.INPUT.GB_K = 21  # Kernel Size (Should be an Odd Number)

    # ====================
    # Dataset CfgNode
    # ====================
    _C.DATASET = CN()
    _C.DATASET.ROOT = ""
    _C.DATASET.NAME = ""
    _C.DATASET.SOURCE_DOMAINS = []
    _C.DATASET.TARGET_DOMAINS = []
    _C.DATASET.SUBSAMPLE_CLASSES = "all"

    # ====================
    # Dataloader CfgNode
    # ====================
    _C.DATALOADER = CN()
    _C.DATALOADER.NUM_WORKERS = 4
    # Setting for the train data loader
    _C.DATALOADER.TRAIN = CN()
    _C.DATALOADER.TRAIN.SAMPLER = "RandomSampler"
    _C.DATALOADER.TRAIN.BATCH_SIZE = 32
    _C.DATALOADER.TRAIN.NUM_INSTANCES = 4
    # Setting for the test data loader
    _C.DATALOADER.TEST = CN()
    _C.DATALOADER.TEST.SAMPLER = "SequentialSampler"
    _C.DATALOADER.TEST.BATCH_SIZE = 32

    # ====================
    # Model CfgNode
    # ====================
    _C.MODEL = CN()
    _C.MODEL.NAME = ""
    _C.MODEL.BACKBONE = ""
    _C.MODEL.METRIC_LOSS_TYPE = "triplet"
    _C.MODEL.NO_MARGIN = False
    _C.MODEL.IF_LABELSMOOTH = "on"
    _C.MODEL.ID_LOSS_WEIGHT = 1.0
    _C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
    _C.MODEL.I2T_LOSS_WEIGHT = 1.0


    # CLIPZeroShot
    _C.MODEL.CLIPZeroShot = CN()
    _C.MODEL.CLIPZeroShot.BACKBONE = "ViT-B/32"

    # SIGLIPZeroShot
    _C.MODEL.SIGLIPZeroShot = CN()
    _C.MODEL.SIGLIPZeroShot.CKPT = "google/siglip-so400m-patch14-384"

    # CLIPLinearProbe
    _C.MODEL.CLIPLinearProbe = CN()
    _C.MODEL.CLIPLinearProbe.BACKBONE = "ViT-B/32"

    # DINOV3ZeroShot
    _C.MODEL.DinoV3ZeroShot = CN()
    _C.MODEL.DinoV3ZeroShot.WEIGHT_PATH = ""
    _C.MODEL.DinoV3ZeroShot.BACKBONE = "dinov3_vitb16"
    _C.MODEL.DinoV3ZeroShot.REPO = ""

    # DINOAdapter
    _C.MODEL.DinoAdapter = CN()
    _C.MODEL.DinoAdapter.WEIGHT_PATH = ""
    _C.MODEL.DinoAdapter.BACKBONE = "dinov3_vitb16"
    _C.MODEL.DinoAdapter.REPO = "" 

    # DinoDistill
    _C.MODEL.DinoDistill = CN()
    _C.MODEL.DinoDistill.WEIGHT_PATH = ""
    _C.MODEL.DinoDistill.BACKBONE = "dinov3_vitb16"
    _C.MODEL.DinoDistill.REPO = ""
    _C.MODEL.DinoDistill.STUDENT_WEIGHT_PATH = ""
    _C.MODEL.DinoDistill.STUDENT_BACKBONE = "dinov3_vitb16"
    _C.MODEL.DinoDistill.DISTILL = False
    _C.MODEL.DinoDistill.FINETUNE = False


    # ====================
    # Optimizer CfgNode
    # ====================
    _C.OPTIM = CN()
    _C.OPTIM.NAME = "sgd"
    _C.OPTIM.LR = 0.0002
    _C.OPTIM.WEIGHT_DECAY = 5e-4
    _C.OPTIM.MOMENTUM = 0.9
    _C.OPTIM.SGD_DAMPENING = 0
    _C.OPTIM.SGD_NESTEROV = False
    _C.OPTIM.ADAM_BETA1 = 0.9
    _C.OPTIM.ADAM_BETA2 = 0.999
    _C.OPTIM.LR_SCHEDULER = "Cosine"
    _C.OPTIM.STEP_SIZE = -1
    _C.OPTIM.GAMMA = 0.1  # Factor to reduce learning rate
    _C.OPTIM.MAX_EPOCH = 10
    _C.OPTIM.WARMUP_EPOCH = (
        -1  # Set WARMUP_EPOCH larger than 0 to activate warmup training
    )
    _C.OPTIM.WARMUP_TYPE = "linear"  # Either linear or constant
    _C.OPTIM.WARMUP_CONS_LR = 1e-5  # Constant learning rate when WARMUP_TYPE=constant
    _C.OPTIM.WARMUP_MIN_LR = 1e-5  # Minimum learning rate when WARMUP_TYPE=linear

    # Domain-specific learning rate configurations, default is to use the same learning rate and scheduler for all domains
    _C.OPTIM.DOMAIN_OPTIM = CN()
    _C.OPTIM.DOMAIN_OPTIM.DOMAIN_LR_MULTIPLIERS = []
    _C.OPTIM.DOMAIN_OPTIM.DOMAIN_SCHEDULERS = []
    _C.OPTIM.DOMAIN_OPTIM.DOMAIN_STEP_SIZES = []

    # ====================
    # Train CfgNode
    # ====================
    _C.TRAIN = CN()
    _C.TRAIN.PRINT_FREQ = 10

    # ====================
    # Solver CfgNode
    # ====================
    _C.SOLVER = CN()
    _C.SOLVER.MARGIN = None

    # ====================
    # Test CfgNode
    # ====================
    _C.TEST = CN()
    _C.TEST.EVALUATOR = "R1_mAP"
    _C.TEST.SPLIT = "Test"
    _C.TEST.FINAL_Model = "last_step"
    # If true, restrict each query to compare only with gallery samples
    # from the same domain (uses domain labels collected from dataloader)
    _C.TEST.SAME_DOMAIN_GALLERY = False

    return _C
