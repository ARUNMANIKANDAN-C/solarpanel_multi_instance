
import  albumentations  as  A
from  albumentations.pytorch  import  ToTensorV2
import  logging
from  typing  import  List,  Dict,  Any

class  AdaptiveAugmentation:
        """
        Applies  stronger  augmentation  to  images  containing  minority  classes.
        Includes  "Crop  Cut"  (CoarseDropout)  for  minority  classes.
        """
        
        def  __init__(self,  config:  Any,  class_weights:  Dict[str,  float]):
                self.config  =  config
                self.class_weights  =  class_weights
                self.logger  =  logging.getLogger(__name__)
                
                self.minority_classes  =  [
                        cls  for  cls,  weight  in  class_weights.items()  
                        if  weight  >  1.2  and  cls  in  class_weights
                ]
                
                self.majority_classes  =  [
                        cls  for  cls,  weight  in  class_weights.items()  
                        if  weight  <  0.8  and  cls  in  class_weights
                ]
                
        def  get_transforms(self,  image_id:  str,  class_distribution:  List[str])  ->  A.Compose:
                contains_minority  =  any(cls  in  self.minority_classes  for  cls  in  class_distribution)
                contains_majority  =  any(cls  in  self.majority_classes  for  cls  in  class_distribution)
                
                #  Base  transforms
                transforms  =  [
                        A.OneOf([
                                A.RandomSizedBBoxSafeCrop(
                                        height=self.config.img_size,  
                                        width=self.config.img_size,  
                                        p=0.5  if  contains_minority  else  0.3
                                ),
                                A.Resize(height=self.config.img_size,  width=self.config.img_size)
                        ],  p=1.0),
                        A.Resize(height=self.config.img_size,  width=self.config.img_size,  p=1.0),
                        A.HorizontalFlip(p=0.5),
                ]
                
                if  contains_minority:
                        #  STRONG  augmentation  with  "Crop  Cut"  (CoarseDropout)
                        transforms.extend([
                                A.VerticalFlip(p=0.3),
                                A.CoarseDropout(
                                        max_holes=8,
                                        max_height=32,
                                        max_width=32,
                                        min_holes=1,
                                        min_height=8,
                                        min_width=8,
                                        fill_value=0,  
                                        p=0.5  #  50%  chance  for  "Crop  Cut"  on  minority  images
                                ),
                                A.ColorJitter(brightness=0.3,  contrast=0.3,  saturation=0.3,  hue=0.1,  p=0.8),
                                A.GaussNoise(var_limit=(10.0,  50.0),  p=0.5),
                                A.GaussianBlur(blur_limit=(3,  7),  p=0.4),
                                A.RandomBrightnessContrast(brightness_limit=0.3,  contrast_limit=0.3,  p=0.7),
                                A.CLAHE(p=0.3),
                        ])
                elif  contains_majority:
                        #  LIGHT  augmentation
                        transforms.extend([
                                A.ColorJitter(brightness=0.1,  contrast=0.1,  saturation=0.1,  hue=0.05,  p=0.3),
                                A.RandomBrightnessContrast(brightness_limit=0.1,  contrast_limit=0.1,  p=0.3),
                        ])
                else:
                        #  MEDIUM  augmentation
                        transforms.extend([
                                A.ColorJitter(brightness=0.2,  contrast=0.2,  saturation=0.2,  hue=0.1,  p=0.5),
                                A.GaussNoise(var_limit=(10.0,  30.0),  p=0.3),
                        ])
                
                #  Always  normalize
                transforms.extend([
                        A.Normalize(mean=self.config.mean,  std=self.config.std),
                        ToTensorV2()
                ])
                
                return  A.Compose(
                        transforms,
                        bbox_params=A.BboxParams(
                                format='pascal_voc',
                                min_visibility=0.3,
                                label_fields=['labels']
                        )
                )
