from orca.model.components.resnet_v1 import resnetv1_configs
from orca.model.components.vit_encoders import vit_encoder_configs

encoders = dict()
encoders.update(resnetv1_configs)
encoders.update(vit_encoder_configs)
