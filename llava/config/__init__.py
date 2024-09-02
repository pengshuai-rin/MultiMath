from .dataset_config import *


DataConfig = {
    "Pretrain": [LLaVA_Pretrain],
    "MathPretrain": [LLaVA_Pretrain, Geo170K_align, MultiMath_caption_ZH, MultiMath_caption_EN],
    "FINETUNE": [LLaVA_Instruct],
    "MathFINETUNE": [Geo170K_qa, MathV360K, MultiMath_solution_ZH, MultiMath_solution_EN],
    "VisionTextMathFINETUNE": [Geo170K_qa, MathV360K, MultiMath_solution_ZH, MultiMath_solution_EN, gsm8k_train, math_train, cmath_dev],
}