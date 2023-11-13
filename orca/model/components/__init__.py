from orca.model.components.resnet_v1 import resnetv1_configs
from orca.model.components.simple_encoders import simple_encoder_configs

encoders = dict()
encoders.update(resnetv1_configs)
encoders.update(simple_encoder_configs)
