
import  torch
import  numpy  as  np
from  collections  import  defaultdict,  deque
import  logging
from  rich.console  import  Console
from  rich.logging  import  RichHandler

#  Setup  Rich  Console
console  =  Console()

def  setup_rich_logging(level=logging.INFO):
        """Configures  logging  to  use  RichHandler."""
        logging.basicConfig(
                level=level,
                format="%(message)s",
                datefmt="[%X]",
                handlers=[RichHandler(console=console,  rich_tracebacks=True)]
        )
        return  logging.getLogger("KaggleTrainer")

class  SmoothedValue(object):
        """Track  a  series  of  values  and  provide  access  to  smoothed  values."""
        def  __init__(self,  window_size=20,  fmt=None):
                if  fmt  is  None:
                        fmt  =  "{median:.4f}  ({global_avg:.4f})"
                self.deque  =  list()
                self.total  =  0.0
                self.count  =  0
                self.window_size  =  window_size
                self.fmt  =  fmt

        def  update(self,  value,  n=1):
                self.deque.append(value)
                self.count  +=  n
                self.total  +=  value  *  n
                if  self.window_size  >  0:
                        if  len(self.deque)  >  self.window_size:
                                self.deque.pop(0)

        @property
        def  median(self):
                d  =  torch.tensor(list(self.deque))
                return  d.median().item()

        @property
        def  avg(self):
                d  =  torch.tensor(list(self.deque),  dtype=torch.float32)
                return  d.mean().item()

        @property
        def  global_avg(self):
                return  self.total  /  self.count

        @property
        def  max(self):
                return  max(self.deque)

        @property
        def  value(self):
                return  self.deque[-1]

        def  __str__(self):
                return  self.fmt.format(
                        median=self.median,
                        avg=self.avg,
                        global_avg=self.global_avg,
                        max=self.max,
                        value=self.value
                )

class  MetricLogger(object):
        def  __init__(self,  delimiter="\t"):
                self.meters  =  defaultdict(SmoothedValue)
                self.delimiter  =  delimiter

        def  update(self,  **kwargs):
                for  k,  v  in  kwargs.items():
                        if  isinstance(v,  torch.Tensor):
                                v  =  v.item()
                        assert  isinstance(v,  (float,  int))
                        self.meters[k].update(v)

        def  __getattr__(self,  attr):
                if  attr  in  self.meters:
                        return  self.meters[attr]
                if  attr  in  self.__dict__:
                        return  self.__dict__[attr]
                raise  AttributeError("'MetricLogger'  object  has  no  attribute  '{}'".format(attr))

        def  __str__(self):
                loss_str  =  []
                for  name,  meter  in  self.meters.items():
                        loss_str.append(
                                "{}:  {}".format(name,  str(meter))
                        )
                return  self.delimiter.join(loss_str)

def  reduce_dict(input_dict,  average=True):
        return  input_dict

def  collate_fn(batch):
        return  tuple(zip(*batch))
