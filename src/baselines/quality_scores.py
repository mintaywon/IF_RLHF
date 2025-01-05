import torch
import datasets
from utils.quality_scores import get_logits, compute_quality_scores
from utils import prepare_model
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig


def get_qulity_scores(
    model_path,
    train_datapath,
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

    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    train_dataset = datasets.load_from_disk(train_datapath)

    logit_list = get_logits(model, train_dataset)
    quality_scores = compute_quality_scores(logit_list)

    return quality_scores
