import os, cv2
# Evito conflitti con thread ed errori (ME NE DAVA ABBASTANZA, HO PROVATO TANTE COSE)
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_DATASETS_DISABLE_MULTIPROCESSING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from datasets import load_from_disk
import matplotlib.pyplot as plt
import numpy as np

model_finetuned = ViTForImageClassification.from_pretrained("./moneta", low_cpu_mem_usage=True)
featur_extractor = ViTImageProcessor.from_pretrained("./moneta")
ds = load_from_disk("./moneta_dataset")

def attention(model, pil, title=""):
    """Prende un modelo e un'immagine PIL e visualizza l'attention map sovrapposta. LO stesso dell'altro File"""
    model.eval()

    inputs = featur_extractor(images=pil, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions

    att_mat = torch.stack(attentions).squeeze(1)
    att_mat = torch.mean(att_mat, dim=1)

    residual_att = torch.eye(att_mat.size(1)).to(device)
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    joint_attentions = torch.zeros(aug_att_mat.size()).to(device)
    joint_attentions[0] = aug_att_mat[0]
    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))

    mask = v[0, 1:].reshape(grid_size, grid_size).cpu().detach().numpy()
    mask = cv2.resize(mask / mask.max(), pil.size)

    image_np = np.array(pil.convert("RGB"))
    plt.figure(figsize=(6, 6))
    plt.imshow(image_np)
    plt.imshow(mask, cmap="jet", alpha=0.45)
    plt.title(f"Attention Map - {title}")
    plt.axis("off")

    plt.show()
    
def inferenza(img_num: int):
    
    img = ds['test'][img_num]['image']

    inputs = featur_extractor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model_finetuned(**inputs)
    logits = outputs.logits

    labels_map = model_finetuned.config.id2label
    probs = torch.nn.Softmax(dim=-1)(logits)
    top_k = min(5, len(labels_map))
    top_predictions = torch.argsort(probs, dim=-1, descending=True)

    print(f"\nPREDIZIONI SU IMMAGINE {img_num}:")
    for idx in top_predictions[0, :top_k]:
        label_id = idx.item()
        print(label_id)
        print(f'{probs[0, label_id]:.5f} : {labels_map[label_id]}')
        
    top_id = top_predictions[0, 0].item()
    print(f"\nRISULTATO PIÙ ALTO: {probs[0, top_id]:.5f}, {labels_map[top_id]}")
    attention(model_finetuned, img, f"INFERENZA: {labels_map[top_id]}")
    return labels_map[top_id]

if __name__ == "__main__":
    inferenza(img_num = 0)