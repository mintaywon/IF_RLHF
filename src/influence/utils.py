import torch
import datasets
import numpy as np
from torch import nn
from transformers import AutoTokenizer
from peft import get_peft_model, PeftModel
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(examples):
    batch = {}
    batch["input_ids_chosen"] = [torch.tensor(ex["input_ids_chosen"]) for ex in examples]
    batch["attention_mask_chosen"] = [torch.tensor(ex["attention_mask_chosen"]) for ex in examples]
    batch["input_ids_rejected"] = [torch.tensor(ex["input_ids_rejected"]) for ex in examples]
    batch["attention_mask_rejected"] = [torch.tensor(ex["attention_mask_rejected"]) for ex in examples]
    batch["input_ids_chosen"] = torch.stack(batch["input_ids_chosen"]).to(device)
    batch["attention_mask_chosen"] = torch.stack(batch["attention_mask_chosen"]).to(device)
    batch["input_ids_rejected"] = torch.stack(batch["input_ids_rejected"]).to(device)
    batch["attention_mask_rejected"] = torch.stack(batch["attention_mask_rejected"]).to(device)
    return batch

def compute_loss(model, inputs):
    model.zero_grad()
    rewards_chosen = model(
        input_ids=inputs["input_ids_chosen"],
        attention_mask=inputs["attention_mask_chosen"],
        return_dict=True,
    )["logits"]
    rewards_rejected = model(
        input_ids=inputs["input_ids_rejected"],
        attention_mask=inputs["attention_mask_rejected"],
        return_dict=True,
    )["logits"]
    loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
    return loss


def prepare_model(model, device, peft_model_id):
    # get_peft_model 
    print("Loading model")
    model = PeftModel.from_pretrained(model, peft_model_id)
    # turn lora weights require_grad to True
    model.to(device)
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        elif 'modules_to_save.default' in name:
            param.requires_grad = True
            print(f"requires_grad of {name} is set to True")
    model.eval()
    model.print_trainable_parameters()
    
    # # modules_to_save is not training, set to training. this if for computing activation on forward hook of opacus
    # for m in model.modules():
    #     if any(p is not None for p in m.parameters(recurse=False)) and any(p.requires_grad for p in m.parameters()):
    #         m.training = True
    return model


def get_dataset(data_path, tokenizer_path):
    train_dataset = datasets.load_from_disk(f"{data_path}/train_dataset")
    eval_dataset = datasets.load_from_disk(f"{data_path}/eval_dataset")
    # pad the examples to max length and return the dataset with padded examples
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast = True)
    train_dataset = train_dataset.map(lambda x: _pad_examples(x, tokenizer))
    eval_dataset = eval_dataset.map(lambda x: _pad_examples(x, tokenizer))
    return train_dataset, eval_dataset

def get_dataset_long_short(data_path, tokenizer_path):
    train_dataset = datasets.load_from_disk(f"{data_path}/train_dataset")
    eval_dataset = datasets.load_from_disk(f"{data_path}/eval_dataset")
    longer_chosen = []
    longer_rejected = []
    # Split data based on the condition
    for example in eval_dataset:
        assert isinstance(example, dict), f"Type of example is {type(example)} and not dict"
        if len(example["input_ids_chosen"]) > len(example["input_ids_rejected"]):
            longer_chosen.append(example)
        else:
            longer_rejected.append(example)
    print(f"{len(longer_chosen)} examples have chosen responses longer than rejected responses")
    print(f"{len(longer_rejected)} examples have rejected responses longer than chosen responses")

    # Convert lists of dictionaries to datasets
    eval_dataset_longer = datasets.Dataset.from_dict({key: [ex[key] for ex in longer_chosen] for key in longer_chosen[0]})
    eval_dataset_shorter = datasets.Dataset.from_dict({key: [ex[key] for ex in longer_rejected] for key in longer_rejected[0]})
    
    # pad the examples to max length and return the dataset with padded examples
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast = True)
    train_dataset = train_dataset.map(lambda x: _pad_examples(x, tokenizer))
    eval_dataset_longer = eval_dataset_longer.map(lambda x: _pad_examples(x, tokenizer))
    eval_dataset_shorter = eval_dataset_shorter.map(lambda x: _pad_examples(x, tokenizer))
    
    # Save datasets to disk
    eval_dataset_longer.save_to_disk(f"{data_path}/eval_dataset_longer_chosen")
    eval_dataset_shorter.save_to_disk(f"{data_path}/eval_dataset_longer_rejected")
    return train_dataset, eval_dataset_longer, eval_dataset_shorter

def _pad_examples(example, tokenizer, max_length=1024):
    # define pad_tokens for GPT-2 on both tokenizer and model, this is specific to GPT-2
    tokenizer.pad_token_id = tokenizer.eos_token_id
    example["input_ids_chosen"] = example["input_ids_chosen"] + [tokenizer.pad_token_id] * (max_length - len(example["input_ids_chosen"]))
    example["attention_mask_chosen"] = example["attention_mask_chosen"] + [0] * (max_length - len(example["attention_mask_chosen"]))
    example["input_ids_rejected"] = example["input_ids_rejected"] + [tokenizer.pad_token_id] * (max_length - len(example["input_ids_rejected"]))
    example["attention_mask_rejected"] = example["attention_mask_rejected"] + [0] * (max_length - len(example["attention_mask_rejected"]))    
    return example

def get_longer_chosen_rejected(eval_dataset, store_path):
    # Initialize lists to hold the entire example dictionaries
    longer_chosen = []
    longer_rejected = []

    # Split data based on the condition
    for example in eval_dataset:
        assert isinstance(example, dict), f"Type of example is {type(example)} and not dict"
        if len(example["input_ids_chosen"]) > len(example["input_ids_rejected"]):
            longer_chosen.append(example)
        else:
            longer_rejected.append(example)
            
    print(f"{len(longer_chosen)} examples have chosen responses longer than rejected responses")
    print(f"{len(longer_rejected)} examples have rejected responses longer than chosen responses")

    # Convert lists of dictionaries to datasets
    eval_dataset_longer_chosen = datasets.Dataset.from_dict({key: [ex[key] for ex in longer_chosen] for key in longer_chosen[0]})
    eval_dataset_longer_rejected = datasets.Dataset.from_dict({key: [ex[key] for ex in longer_rejected] for key in longer_rejected[0]})

    # Save datasets to disk
    eval_dataset_longer_chosen.save_to_disk(f"{store_path}/eval_dataset_longer_chosen")
    eval_dataset_longer_rejected.save_to_disk(f"{store_path}/eval_dataset_longer_rejected")
    
    return eval_dataset_longer_chosen, eval_dataset_longer_rejected


def rapid_datainf(rapid_grad_train, rapid_grad_val, indices):
    n_train = len(rapid_grad_train)
    
    # calculate lambda
    lam = 0
    for grad in rapid_grad_train:
        lam += torch.mean(grad**2)
    lam = 0.1 / n_train * lam
    # calculate avg gradient of validation set
    val_grad = torch.zeros_like(rapid_grad_val[0])
    for i, grad in enumerate(rapid_grad_val):
        if i in indices:
            val_grad += grad
    val_grad /= len(indices)

    # make a tensor of shape (n_train, D) where each row is a flattened gradient
    train_grads = torch.stack(rapid_grad_train)
    train_grads_dots = torch.matmul(train_grads, train_grads.t()) # this stores dots of all pairs of gradients in train_grads
    val_grad_dots = torch.matmul(train_grads, val_grad.t()) # this stores dots of train_grads with val_grad_avg
    # Initialize inf_list as a tensor for better performance
    rapidinf = torch.zeros(n_train)

    # Calculate the first term outside the loop
    rapidinf = -1 / lam * val_grad_dots

    # Precompute terms
    one_over_lam = 1 / lam
    one_over_lam_n_train = one_over_lam / n_train
    lam_plus_diag = lam + train_grads_dots.diag()

    # Use vectorized operations for the second term
    for k in range(n_train):
        rapidinf[k] += torch.sum(one_over_lam_n_train * (train_grads_dots[:, k] * val_grad_dots) / lam_plus_diag)
    
    return rapidinf.tolist()

def rapid_tracin(rapid_grad_train, rapid_grad_val, indices):
    n_train = len(rapid_grad_train)
    
    # calculate lambda
    lam = 0
    for grad in rapid_grad_train:
        lam += torch.mean(grad**2)
    lam = 0.1 / n_train * lam
    
    # calculate avg gradient of validation set
    val_grad = torch.zeros_like(rapid_grad_val[0])
    for i, grad in enumerate(rapid_grad_val):
        if i in indices:
            val_grad += grad
    val_grad /= len(indices)

    # make a tensor of shape (n_train, D) where each row is a flattened gradient
    train_grads = torch.stack(rapid_grad_train)
    val_grad_dots = torch.matmul(train_grads, val_grad.t()) # this stores dots of train_grads with val_grad_avg

    rapidinf = torch.zeros(n_train)
    rapidinf = -1 / lam * val_grad_dots
    return rapidinf.tolist()

def rapid_selfinf(rapid_grad_train):
    train_grads = torch.stack(rapid_grad_train)
    train_grads_dots = torch.matmul(train_grads, train_grads.t()) # this stores dots of all pairs of gradients in train_grads
    
    return train_grads_dots.diag().tolist()
    

def get_sycophancy_indices(data):
    A_indices, B_indices, C_indices, D_indices = [], [], [], []
    for i, example in tqdm(enumerate(data)):
        if example['answer_gpt'] == 1:
            A_indices.append(i)
        elif example['answer_gpt'] == 2:
            B_indices.append(i)
        elif example['answer_gpt'] == 3:
            C_indices.append(i)
        elif example['answer_gpt'] == 4:
            D_indices.append(i)
        else:
            print(f"Invalid answer at index {i}: {example['message_gpt']}\n\n\n")
    return A_indices, B_indices, C_indices, D_indices

def get_length_indices(data):
    shorter_indices, longer_indices = [], []
    for i, example in tqdm(enumerate(data)):
        if len(example['input_ids_chosen']) < len(example['input_ids_rejected']):
            shorter_indices.append(i)  
        else:
            longer_indices.append(i)
    return shorter_indices, longer_indices

def get_val_fpr_tpr(val_data, flipped_indices):
    fn, tn, fp, tp = 0, 0, 0, 0
    for i, data in enumerate(val_data):
        if i in flipped_indices:
            if data['answer_llm'] == 1:
                tp += 1
            elif data['answer_llm'] == 2:
                fn += 1
        else:
            if data['answer_llm'] == 1:
                tn += 1
            elif data['answer_llm'] == 2:
                fp += 1
    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)   
    return fpr, tpr

def get_roc_auc(influence, flipped_indices):
    total_points = len(influence)
    x_vals = np.linspace(0, 1, total_points)

    # Generate the true conditions assuming that the detection of flipped indices is binary
    true_conditions = np.zeros_like(x_vals)

    for i in range(total_points):
        if i in flipped_indices:
            true_conditions[i] = 1
    
    fpr, tpr, _ = roc_curve(true_conditions, influence)
    # Calculate the AUC
    roc_auc = auc(fpr, tpr)
    return roc_auc, fpr, tpr

def plot_roc_curve(influence, flipped_indices, title, fpr_llm = [], tpr_llm = [], llm_label = []):
    """
    Plots the ROC curve for given data and flipped indices and calculates the AUC value.

    Parameters:
    influence (np.array): Array of data points from the RapidInf algorithm.
    flipped_indices (list): List of indices that were flipped.
    noise_percentage (int): The percentage of noise used in the title of the plot. Default is 20%.

    Returns:
    float: AUC value of the ROC curve.
    """
    roc_auc, fpr, tpr = get_roc_auc(influence, flipped_indices)
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    # Scatter LLM points, checking that the lists have the same length
    if len(fpr_llm) == len(tpr_llm) == len(llm_label):
        if len(fpr_llm) > 0:
            for fpr, tpr, label in zip(fpr_llm, tpr_llm, llm_label):
                plt.scatter(fpr, tpr, lw=2, label=label)
    else:
        print("Warning: fpr_llm, tpr_llm, and llm_label must be lists of the same length.")
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.title(title)
    plt.show()
    print("AUC value:", roc_auc)
    return roc_auc

def calculate_roc_auc(rapid_grad_train, rapid_grad_val, flipped_indices, subset_indices, N, S = 0):
    """
    Calculates the ROC AUC value for the given influence data, flipped indices, and subset indices.

    Parameters:
    influence (np.array): Array of data points from the RapidInf algorithm.
    flipped_indices (list): List of indices that were flipped.
    subset_indices (np.array): 2D numpy array of random indices of shape (N, S).

    Returns:
    float: AUC value of the ROC curve.
    """
    n_val = len(rapid_grad_val)
    if S == 0:
        S = n_val // 2 # subset size of eval set, use half of the eval set
    import random
    random.seed(42)
    import numpy as np
    from tqdm import tqdm
    # generate random indices from 0 to n_val - 1 with size S, N times
    random_indices = np.random.randint(0, n_val, (N, S))
    # Generate random indices from 0 to n_val - 1 with size S, N times
    random_indices = [random.sample(range(n_val), S) for _ in range(N)]
    # Convert to numpy array for easier handling
    random_indices_np = np.array(random_indices)
    print(random_indices_np.shape) # (N, S)
   
    # Generate the true conditions assuming that the detection of flipped indices is binary
    n_train = len(rapid_grad_train)
    true_conditions = np.zeros(len(rapid_grad_train))

    for i in range(len(rapid_grad_train)):
        if i in flipped_indices:
            true_conditions[i] = 1
            
    # Initialize the list to store AUC values
    auc_list = []
    lam = 0
    for grad in rapid_grad_train:
        lam += torch.mean(grad**2)
    lam = 0.1 / n_train * lam
    # make a tensor of shape (n_train, D) where each row is a flattened gradient
    train_grads = torch.stack(rapid_grad_train)
    # compute train_grads * train_grads^T
    train_grads_dots = torch.matmul(train_grads, train_grads.t()) # this stores dots of all pairs of gradients in train_grads
    # Precompute terms that can be reused
    one_over_lam = 1 / lam
    one_over_lam_n_train = one_over_lam / n_train
    lam_plus_diag = lam + train_grads_dots.diag()
    for indices in tqdm(subset_indices):
        ### Caluclate RapidInf
        val_grad = torch.zeros_like(rapid_grad_val[0])
        for i, grad in enumerate(rapid_grad_val):
            if i in indices:
                val_grad += grad
        val_grad /= len(indices)
        # compute train_grads * val_grad_avg^T
        val_grad_dots = torch.matmul(train_grads, val_grad.t()) # this stores dots of train_grads with val_grad_avg
        # Initialize inf_list as a tensor for better performance
        rapidinf = torch.zeros(n_train)
        # Calculate the first term outside the loop
        rapidinf = -1 / lam * val_grad_dots
        # Use vectorized operations for the second term
        for k in range(n_train):
            rapidinf[k] += torch.sum(one_over_lam_n_train * (train_grads_dots[:, k] * val_grad_dots) / lam_plus_diag)
        influence = rapidinf.tolist()
        
        
        ### Calculate ROC AUC
        fpr, tpr, _ = roc_curve(true_conditions, influence)
        # Calculate the AUC
        roc_auc = auc(fpr, tpr)
        auc_list.append(roc_auc)
    
    return auc_list