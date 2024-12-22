import torch
import argparse
import datasets
import os
import pickle
import torch.nn.functional as F
import numpy as np

from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
from torch.utils.data import DataLoader
from influence.utils import collate_fn, prepare_model
from influence.main_influence import compute_lambda
from influence.utils import compute_loss, compute_loss_negative, compute_loss_positive, compute_loss_positive_rejected
from tqdm import tqdm
from copy import copy

class RapidGrad():
    def __init__(self, shuffle_lambda, filepath, device, seed=42):
        self.is_init = False
        self.D = None
        self.K = None
        self.random_mat = None
        self.M = 1
        self.shuffle_lambda = shuffle_lambda
        self.perm_mat_list = []
        self.perm_dim_list = []
        self.filepath = filepath
        self.device = device
        self.seed = seed

    def __call__(self, vec, K):
        if self.is_init == False:
            print("Creating random and shuffling matrices. It may take a few minutes.")
            D = len(vec)
            self.init(D)
        for i, (dim, perm_mat) in enumerate(zip(self.perm_dim_list, self.perm_mat_list)):
            if i%2 == 0:
                vec = vec.reshape((dim, -1))
                vec = vec[perm_mat, :]
            else:
                vec = vec.reshape((-1, dim))
                vec = vec[:, perm_mat]
        vec = vec.reshape((-1))
        vec = vec*self.random_mat

        if isinstance(K, list):
            vec_list = []
            for k in K:
                step = self.D//k
                vec_list.append(torch.sum(vec.reshape((-1, step)), axis=1))
            return vec_list
        else:
            step = self.D//K
            vec = torch.sum(vec.reshape((-1, step)), axis=1)
            return vec

    def init(self, D):
        self.is_init = True
        np.random.seed(self.seed)
        self.D = D
        self.file_name = os.path.join(
            self.filepath,
            f"RapidGrad_D{self.D}_n{self.shuffle_lambda}_seed{self.seed}.obj"
        )
        if not self.load():
            self.create_random_mat(D)
            self.create_perm_mat(D)
            self.save()
        self.random_mat = torch.from_numpy(self.random_mat).to(dtype=torch.float16).to(self.device)

    def create_random_mat(self, D):
        self.random_mat = np.random.randint(0, 2, (D,), dtype=np.int8)
        self.random_mat[self.random_mat < 1e-8] = -1

    def create_perm_mat(self, D):
        lt = []
        while D != 1:
            for i in range(2, int(D + 1)):
                if D % i == 0:
                    lt.append(i)
                    D = D / i
                    break
        for _ in tqdm(range(self.shuffle_lambda)):
            x = np.random.randint(len(lt)//4, len(lt)//2 + 1)
            np.random.shuffle(lt)
            dim = np.prod(lt[:x], dtype=np.longlong)
            self.perm_dim_list.append(dim)
            self.perm_mat_list.append(np.random.permutation(dim))

    def save(self):
        if os.path.exists(self.file_name):
            return
        with open(self.file_name, 'wb') as f:
            pickle.dump(self, f)

    def load(self):
        if not os.path.exists(self.file_name):
            return False
        with open(self.file_name, 'rb') as f:
            new_obj = pickle.load(f)
        device = self.device
        self.__dict__ = copy(new_obj.__dict__)
        self.device = device
        return True

def pad(x):
    D = len(x)
    K = 2**24
    new_D = ((D - 1)//K + 1)*K
    x = F.pad(x, (0, new_D - D), "constant", 0)
    return x

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--model_path", type=str, default="")
    
    # data
    parser.add_argument("--data_path", type=str, default="")
    
    # save name
    parser.add_argument("--save_name", type=str, default="")
    
    # quantization
    parser.add_argument("--load_in_4bit", type=bool, default=True)
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
    parser.add_argument("--use_bnb_nested_quant", type=bool, default=True)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--train_batchsize", type=int, default=1)
    parser.add_argument("--eval_batchsize", type=int, default=1)
    
    # oporp
    parser.add_argument("--shuffle_lambda", type=int, default=100)
    parser.add_argument("--K", type=lambda x: list(map(int, x.split(','))), default=[2**16], help="List of integers of powers of 2")
    parser.add_argument("--seed", type=int, default=42)
    
    #loss type
    parser.add_argument("--loss_type", type=str, default="preference")
    
    return parser.parse_args()



if __name__ == '__main__':
    args = parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.load_in_4bit:
        compute_dtype = torch.float16
        if args.torch_dtype not in {"auto", None}:
            compute_dtype = getattr(torch, args.torch_dtype)
        print(compute_dtype)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.use_bnb_nested_quant,
        )
    else:
        quantization_config = None
    
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=1, quantization_config=quantization_config)
    # turn the lora weights to require_grad, to compute gradients and send model to device
    model = prepare_model(model, device, args.model_path)
    # get dataset retrieves the dataset after padding the dataset to max_length
    train_dataset = datasets.load_from_disk(args.data_path)
    # train_dataset, eval_dataset = get_dataset(args.model_path, args.tokenizer_path)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    # create a loss function mapping
    loss_function_mapping = {
        'preference': compute_loss,
    }
    loss_function = loss_function_mapping.get(args.loss_type)
    
    #####
    # step 2. compute the gradients for each training data point, and cache them using OPORP
    #####
    oporp = RapidGrad(args.shuffle_lambda, args.model_path, device, args.seed)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn = collate_fn)
    rapid_grad_dict = {key: [] for key in args.K}
    for batch in tqdm(train_dataloader):
        model.zero_grad()
        loss = loss_function(model, batch)
        loss.backward()
        grad_list = []
        for k, v in model.named_parameters():
            if 'lora_A' in k or 'modules_to_save.default' in k:
                grad_list.append(v.grad.reshape(-1))
            elif 'lora_B' in k:
                # first index of shape indicates low-rank
                grad_list.append(v.grad.reshape(-1))
            else:
                pass
        grad_vec = torch.cat(grad_list) # flatten the gradients into a single vector
        # pad the grad_vec
        grad_vec = pad(grad_vec)
        train_vecs = oporp(grad_vec, args.K) # this is a list of length len(K) of the OPORP vectors
        train_vecs = [train_vec.cpu() for train_vec in train_vecs]
        for i, train_vec in enumerate(train_vecs):
            rapid_grad_dict[args.K[i]].append(train_vec)
        
    # save train_vec_dict as pt
    torch.save(rapid_grad_dict, args.model_path + f"/{args.save_name}")
    
