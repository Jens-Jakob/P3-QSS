#guide https://github.com/TylerYep/torchinfo
from torchinfo import summary
import feature_extractor

#Modellen her!
model = feature_extractor.build_feature_extractor()
summary(model, (1, 3, 224, 224), depth=3)