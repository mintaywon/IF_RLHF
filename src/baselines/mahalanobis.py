import torch
import datasets
from utils.mahalanobis import get_activations, estimate_mahalanobis_score_using_activations
from utils import prepare_model
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig


def get_mahalanobis_score(
    model_path,
    train_datapath,
    eval_datapath,
    load_in_4bit=True,
    torch_dtype="bfloat16",
    bnb_4bit_quant_type="nf4",
    use_bnb_nested_quant=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if load_in_4bit:
        compute_dtype = torch.float16
        if torch_dtype not in {"auto", None}:
            compute_dtype = getattr(torch, torch_dtype)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=use_bnb_nested_quant,
        )
    else:
        quantization_config = None

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=1, quantization_config=quantization_config
    )
    model = prepare_model(model, device, model_path)

    # Load dataset
    eval_dataset = datasets.load_from_disk(eval_datapath)
    train_dataset = datasets.load_from_disk(train_datapath)

    # Extract activations
    train_chosen_activations = get_activations(train_dataset, model, device, chosen=True)
    train_rejected_activations = get_activations(train_dataset, model, device, chosen=False)
    eval_chosen_activations = get_activations(eval_dataset, model, device, chosen=True)
    eval_rejected_activations = get_activations(eval_dataset, model, device, chosen=False)

    # Compute mahalanobis distance
    mahalanobis_score = estimate_mahalanobis_score_using_activations(
        eval_chosen_activations,
        eval_rejected_activations,
        train_chosen_activations,
        train_rejected_activations,
    )

    return mahalanobis_score
