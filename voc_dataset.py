
import  torch
from  torch.utils.data  import  Dataset
import  xml.etree.ElementTree  as  ET
import  numpy  as  np
from  PIL  import  Image
import  logging
import  random
from  collections  import  Counter
import  albumentations  as  A
from  albumentations.pytorch  import  ToTensorV2
from  typing  import  Optional,  List,  Dict,  Tuple,  Union,  Any
from  rich.table  import  Table
from  rich.console  import  Console
from  voc_aug  import  AdaptiveAugmentation

console  =  Console()

class  VOCDataset(Dataset):
        def  __init__(
                self,
                config:  Any,
                image_set:  str,
                image_ids:  Optional[List[str]]  =  None,
                transforms:  Optional[Union[A.Compose,  AdaptiveAugmentation]]  =  None,
                class_list:  Optional[List[str]]  =  None,
                logger:  Optional[logging.Logger]  =  None
        ):
                self.config  =  config
                self.image_set  =  image_set
                self.transforms  =  transforms
                self.logger  =  logger  or  logging.getLogger(__name__)
                
                #  Load  image  IDs
                if  image_ids  is  not  None:
                        self.image_ids  =  image_ids
                else:
                        self.image_ids  =  self._load_image_ids()
                
                #  Setup  classes
                if  class_list:
                        self.class_list  =  class_list
                else:
                        self.class_list  =  self._determine_classes()
                
                self.class_map  =  {name:  idx  for  idx,  name  in  enumerate(self.class_list,  1)}
                self.inv_class_map  =  {v:  k  for  k,  v  in  self.class_map.items()}
                
                #  Filter  and  stats
                self.image_ids,  self.class_distribution  =  self._filter_and_stats()
                self.class_weights  =  self._calculate_class_weights()
                self.minority_classes  =  [cls  for  cls,  w  in  self.class_weights.items()  if  w  >  1.2]
                self.majority_classes  =  [cls  for  cls,  w  in  self.class_weights.items()  if  w  <  0.8]
                
                if  isinstance(transforms,  AdaptiveAugmentation):
                        self.adaptive_aug  =  transforms
                        self.regular_transforms  =  None
                else:
                        self.adaptive_aug  =  None
                        self.regular_transforms  =  transforms

                self._log_init()

        def  _log_init(self):
                self.logger.info(f"Initialized  {self.image_set}  set  with  {len(self)}  images")
                
                if  self.class_distribution:
                        #  Create  a  Rich  Table  for  distribution
                        table  =  Table(title=f"Class  Distribution  ({self.image_set})",  show_header=True,  header_style="bold  magenta")
                        table.add_column("Class",  style="cyan")
                        table.add_column("Count",  justify="right",  style="green")
                        table.add_column("Percentage",  justify="right",  style="yellow")
                        
                        sorted_classes  =  sorted(self.class_distribution.items(),  key=lambda  x:  x[1],  reverse=True)
                        total  =  sum(self.class_distribution.values())
                        
                        for  cls,  count  in  sorted_classes:
                                percentage  =  (count  /  total)  *  100  if  total  >  0  else  0
                                table.add_row(cls,  str(count),  f"{percentage:.1f}%")
                                
                        #  Print  table  to  console  directly  since  logger  might  mangle  table  structure
                        console.print(table)
                        #  Also  log  a  summary  line  for  file  logs
                        self.logger.info(f"Top  3  classes:  {',  '.join([f'{c}:  {n}'  for  c,  n  in  sorted_classes[:3]])}")

        def  _format_class_distribution(self)  ->  str:
                #  Kept  for  compatibility  if  needed  elsewhere,  but  mainly  replaced  by  _log_init  table
                if  not  self.class_distribution:
                        return  "    No  classes  found"
                sorted_classes  =  sorted(self.class_distribution.items(),  key=lambda  x:  x[1],  reverse=True)
                total  =  sum(self.class_distribution.values())
                return  '\n'.join([f"    {cls}:  {count}  ({count/total*100:.1f}%)"  for  cls,  count  in  sorted_classes])

        def  _load_image_ids(self)  ->  List[str]:
                split_file  =  self.config.splits_dir  /  f"{self.image_set}.txt"
                if  split_file.exists():
                        with  open(split_file,  'r')  as  f:
                                return  [line.strip().split()[0]  for  line  in  f  if  line.strip()]
                else:
                        image_files  =  sorted(self.config.images_dir.glob("*.jpg"))
                        return  [f.stem  for  f  in  image_files]
        
        def  _determine_classes(self)  ->  List[str]:
                classes  =  set()
                sample_size  =  min(500,  len(self.image_ids))
                sample_ids  =  random.sample(self.image_ids,  sample_size)
                for  img_id  in  sample_ids:
                        try:
                                xml_path  =  self.config.annotations_dir  /  f"{img_id}.xml"
                                if  xml_path.exists():
                                        tree  =  ET.parse(xml_path)
                                        for  obj  in  tree.findall("object"):
                                                classes.add(obj.find("name").text)
                        except:  continue
                return  sorted(classes)
        
        def  _filter_and_stats(self)  ->  Tuple[List[str],  Dict[str,  int]]:
                valid_ids  =  []
                class_counter  =  Counter()
                for  img_id  in  self.image_ids:
                        try:
                                boxes,  labels,  label_names,  _  =  self._parse_annotation(img_id)
                                if  len(boxes)  >  0:
                                        for  name  in  label_names:
                                                if  name  in  self.class_map:
                                                        class_counter[name]  +=  1
                                        valid_ids.append(img_id)
                        except:  continue
                return  valid_ids,  dict(class_counter)
        
        def  _calculate_class_weights(self)  ->  Dict[str,  float]:
                if  not  self.class_distribution:  return  {}
                total  =  sum(self.class_distribution.values())
                num_classes  =  len(self.class_distribution)
                return  {cls:  total  /  (count  *  num_classes)  if  count  >  0  else  0  for  cls,  count  in  self.class_distribution.items()}
        
        def  _parse_annotation(self,  image_id:  str)  ->  Tuple[np.ndarray,  np.ndarray,  List[str],  Tuple[float,  float]]:
                xml_path  =  self.config.annotations_dir  /  f"{image_id}.xml"
                if  not  xml_path.exists():  raise  FileNotFoundError(f"Annotation  not  found:  {xml_path}")
                
                try:
                        tree  =  ET.parse(xml_path)
                        root  =  tree.getroot()
                        size  =  root.find("size")
                        width  =  float(size.find("width").text)
                        height  =  float(size.find("height").text)
                        
                        boxes  =  []
                        label_names  =  []
                        for  obj  in  root.findall("object"):
                                if  obj.find("name").text  not  in  self.class_map:  continue
                                bbox  =  obj.find("bndbox")
                                xmin  =  float(bbox.find("xmin").text)
                                ymin  =  float(bbox.find("ymin").text)
                                xmax  =  float(bbox.find("xmax").text)
                                ymax  =  float(bbox.find("ymax").text)
                                
                                xmin  =  max(0,  min(xmin,  width))
                                ymin  =  max(0,  min(ymin,  height))
                                xmax  =  max(xmin,  min(xmax,  width))
                                ymax  =  max(ymin,  min(ymax,  height))
                                
                                if  xmax  >  xmin  and  ymax  >  ymin:
                                        boxes.append([xmin,  ymin,  xmax,  ymax])
                                        label_names.append(obj.find("name").text)
                        
                        boxes  =  np.array(boxes,  dtype=np.float32)  if  boxes  else  np.zeros((0,  4),  dtype=np.float32)
                        labels  =  np.array([self.class_map[name]  for  name  in  label_names],  dtype=np.int64)  if  label_names  else  np.zeros(0,  dtype=np.int64)
                        return  boxes,  labels,  label_names,  (width,  height)
                except:  raise  ValueError(f"Error  parsing  {xml_path}")

        def  __len__(self):  return  len(self.image_ids)
        
        def  __getitem__(self,  idx):
                image_id  =  self.image_ids[idx]
                img_path  =  self.config.images_dir  /  f"{image_id}.jpg"
                
                try:
                        image  =  np.array(Image.open(img_path).convert("RGB"))
                        boxes,  labels,  label_names,  _  =  self._parse_annotation(image_id)
                        
                        if  self.adaptive_aug:
                                transforms  =  self.adaptive_aug.get_transforms(image_id,  label_names)
                        else:
                                transforms  =  self.regular_transforms
                        
                        if  transforms  and  len(boxes)  >  0:
                                transformed  =  transforms(image=image,  bboxes=boxes,  labels=labels)
                                image  =  transformed["image"]
                                boxes  =  np.array(transformed["bboxes"])  if  transformed["bboxes"]  else  np.zeros((0,  4))
                                labels  =  np.array(transformed["labels"])  if  transformed["labels"]  else  np.zeros(0)
                        elif  not  isinstance(image,  torch.Tensor):  #  Handling  for  no  transforms  case  or  empty  boxes
                                    image  =  ToTensorV2()(image=image)["image"]
                                    image  =  A.Normalize(mean=self.config.mean,  std=self.config.std)(image=image.permute(1,2,0).numpy())["image"]
                                    image  =  torch.from_numpy(image).permute(2,0,1)

                except  Exception  as  e:
                        #  Fallback
                        return  self.__getitem__((idx  +  1)  %  len(self))

                target  =  {
                        "boxes":  torch.as_tensor(boxes,  dtype=torch.float32),
                        "labels":  torch.as_tensor(labels,  dtype=torch.int64),
                        "image_id":  torch.tensor([idx])
                }
                return  image,  target

class  VOCSplitManager:
        def  __init__(self,  config:  Any,  logger:  logging.Logger):
                self.config  =  config
                self.logger  =  logger
                
        def  create_splits(self,  train_ratio=0.7,  val_ratio=0.15,  test_ratio=0.15,  use_adaptive_aug=True):
                full_dataset  =  VOCDataset(self.config,  "trainval")
                all_ids  =  full_dataset.image_ids
                #  Simple  random  split  for  demo  purposes  if  files  don't  exist
                random.shuffle(all_ids)
                n  =  len(all_ids)
                train_end  =  int(n  *  train_ratio)
                val_end  =  int(n  *  (train_ratio  +  val_ratio))
                
                train_ids  =  all_ids[:train_end]
                val_ids  =  all_ids[train_end:val_end]
                test_ids  =  all_ids[val_end:]
                
                #  Create  Datasets
                train_dataset  =  VOCDataset(self.config,  "train",  image_ids=train_ids,  class_list=full_dataset.class_list)
                if  use_adaptive_aug:
                        train_dataset.transforms  =  AdaptiveAugmentation(self.config,  train_dataset.class_weights)
                        train_dataset.adaptive_aug  =  train_dataset.transforms
                        
                val_dataset  =  VOCDataset(
                        self.config,  "val",  image_ids=val_ids,  class_list=full_dataset.class_list,
                        transforms=A.Compose([
                                A.Resize(self.config.img_size,  self.config.img_size),
                                A.Normalize(mean=self.config.mean,  std=self.config.std),
                                ToTensorV2()
                        ],  bbox_params=A.BboxParams(format='pascal_voc',  label_fields=['labels']))
                )
                
                return  {'train':  train_dataset,  'val':  val_dataset}
